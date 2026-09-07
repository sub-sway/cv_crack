"""Dependency-free IMU prototype. Body: X forward, Y left, Z up (right-handed).

World: initial forward/left/up, or magnetic north/west/up after calibration.
Velocity/position are unconstrained inertial integration, NOT wheel odometry.
"""
import math
import time

G = 9.80665


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def spread(values):
    """Population standard deviation, robust where max-min is not."""
    values = list(values)
    if len(values) < 2:
        return 0.
    mean = sum(values)/len(values)
    return math.sqrt(sum((v-mean)**2 for v in values)/len(values))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def conjugate(q):
    return [q[0], -q[1], -q[2], -q[3]]


def rotate(q, v):
    return multiply(multiply(q, [0, *v]), conjugate(q))[1:]


def from_euler(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
            cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def to_euler(q):
    """Return right-handed body roll, pitch and CCW yaw in radians."""
    w, x, y, z = q
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return [roll, pitch, yaw]


def direction8(heading, magnetic=False):
    labels = ['북', '북동', '동', '남동', '남', '남서', '서', '북서'] if magnetic else ['기준', '우전', '우', '우후', '반대', '좌후', '좌', '좌전']
    return labels[int((heading+22.5)//45) % 8]


def vector(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('Expected a three-element vector')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError('Non-finite or nonnumeric vector')
    return [float(x) for x in value]


class Processor:
    def __init__(self, axes=(1, 2, 3), calibration_samples=200, mag_calibration=None,
                 skip_calibration=False):
        # Signed permutation: e.g. (2,-1,3) means sensor Y -> body X.
        if sorted(abs(x) for x in axes) != [1, 2, 3]:
            raise ValueError('axes must be a signed permutation of 1,2,3')
        basis = [[(1 if a > 0 else -1) if abs(a)-1 == i else 0 for i in range(3)] for a in axes]
        if sum(x*y for x, y in zip(cross(basis[0], basis[1]), basis[2])) != 1:
            raise ValueError('Mounting axes must be right-handed')
        self.axes = axes
        self.calibration_samples = calibration_samples
        self.skip_calibration = skip_calibration
        self.mag_calibration = mag_calibration
        if mag_calibration is not None:
            if mag_calibration.get('schema') != 1 or tuple(mag_calibration.get('axes', [])) != tuple(axes):
                raise ValueError('Mag calibration schema/mounting axes mismatch')
            vector(mag_calibration.get('offset_uT'))
            scales = vector(mag_calibration.get('scale'))
            field = mag_calibration.get('field_uT')
            if any(not .1 < s < 10 for s in scales) or type(field) not in (int, float) or not 15 < field < 80:
                raise ValueError('Invalid mag calibration scale/field')
        self.epoch = 0
        self.reset()

    def reset(self):
        self.epoch += 1
        self.previous = None
        self.samples = []
        self.bias = None
        self.startup_accel_norm = G
        self.accel_warning = False
        self.q = [1., 0., 0., 0.]
        self.velocity = [0., 0., 0.]
        self.position = [0., 0., 0.]
        self.previous_linear = [0., 0., 0.]
        self.stationary_samples = 0
        self.magnetic = False
        self.last_mag = None
        self.last_mag_time = 0.
        self.last_mag_accepted = 0.
        self.state = {'schema': 1, 'status': 'disconnected', 'message': 'IMU 연결 대기'}

    def resume(self):
        """Return from a reception gap. Bias and attitude stay; speed and the
        stationary run, unknowable across the blackout, do not."""
        self.velocity = [0., 0., 0.]
        self.previous_linear = [0., 0., 0.]
        self.stationary_samples = 0

    def mounted(self, v):
        return [v[abs(a)-1] * (1 if a > 0 else -1) for a in self.axes]

    def update(self, packet, now=None):
        if not isinstance(packet, dict) or packet.get('v') != 1 or packet.get('type') != 'imu':
            raise ValueError('Unsupported packet')
        t, seq = packet.get('t_us'), packet.get('seq')
        if any(type(x) is not int or not 0 <= x <= 0xffffffff for x in (t, seq)):
            raise ValueError('Invalid device timestamp/sequence')
        if packet.get('chip') not in ('MPU6500', 'MPU9250', 'MPU9255'):
            raise ValueError('Unknown sensor')
        if type(packet.get('mag_present')) is not bool:
            raise ValueError('Invalid magnetometer flag')
        acc = self.mounted(vector(packet.get('accel_mps2')))
        gyro = self.mounted(vector(packet.get('gyro_rps')))
        mag = packet.get('mag_uT')
        mag = self.mounted(vector(mag)) if mag is not None else None
        if norm(acc) > 70 or norm(gyro) > 20:
            raise ValueError('Out-of-range sample')
        dt = 0.
        gap = False
        if self.previous is not None:
            dt = ((t-self.previous[0]) & 0xffffffff)/1e6
            if seq == self.previous[1] or dt == 0:
                raise ValueError('Duplicate sample')
            if ((seq-self.previous[1]) & 0xffffffff) > 0x7fffffff or dt > 60:
                # Sequence or device clock ran backwards: the board rebooted and
                # nothing measured before it describes this device any more.
                self.reset()
                dt = 0.
            elif dt > .1 or ((seq-self.previous[1]) & 0xffffffff) > 20:
                # Only reception dropped out. A loaded host overruns the serial
                # buffer for longer than this on its own, and discarding the
                # gyro bias here restarts the two-second hold every time, which
                # in practice never lets the reading appear at all. The bias is
                # a property of the sensor, not of the pose, so it survives;
                # what cannot survive is dead reckoning across the blackout.
                self.resume()
                gap = True
                dt = 0.
        self.previous = (t, seq)
        now = time.time() if now is None else now
        calibrated_mag = None
        if mag is not None and self.mag_calibration:
            c = self.mag_calibration
            candidate = [(mag[i]-c['offset_uT'][i])*c['scale'][i] for i in range(3)]
            if abs(norm(candidate)-c['field_uT']) < .3*c['field_uT']:
                calibrated_mag = candidate
                self.last_mag, self.last_mag_time = candidate, now
        state = {'schema': 1, 'status': 'calibrating', 'timestamp_unix': now, 'stream_gap': gap,
                 'chip': packet['chip'], 'mag_present': packet['mag_present'], 'mag_uT': mag,
                 'epoch': self.epoch, 'seq': seq, 'heading_reference': 'relative',
                 'heading_deg': None, 'direction8': None, 'experimental': True,
                 'accel_mps2': acc, 'gyro_rps': gyro, 'axes': list(self.axes), 'frame': 'initial_forward_left_up'}
        if self.bias is None and self.skip_calibration:
            # The operator opted out of the stationary hold. Nothing below is
            # measured: the gyro bias is assumed to be zero and the attitude
            # comes from a single accelerometer sample. Heading then drifts at
            # whatever the real bias is, and roll/pitch settle at bias/1.5 rad
            # off, because only the accelerometer pulls them back. Readings stay
            # flagged so nothing here is mistaken for a calibrated measurement.
            self.bias = [0., 0., 0.]
            self.startup_accel_norm = norm(acc) or G
            self.accel_warning = abs(self.startup_accel_norm-G) > .35
            self.q = from_euler(math.atan2(acc[1], acc[2]),
                                math.atan2(-acc[0], math.hypot(acc[1], acc[2])), 0.)
            self.samples.clear()
            dt = 0.
        if self.bias is None:
            # Bound gross motion/faults, then test variation over the ENTIRE window.
            # A constant gyro offset is what calibration needs to estimate, not reject.
            # Slow constant rotation remains indistinguishable from bias: user must stop.
            plausible = norm(gyro) < .3 and abs(norm(acc)-G) < 1.5
            if not plausible:
                self.samples.clear()
            else:
                self.samples.append((acc, gyro))
                self.samples = self.samples[-self.calibration_samples:]
            stable = False
            if self.samples:
                # Variation is still measured over the ENTIRE window, but as a
                # standard deviation rather than max-min. Peak-to-peak is set by
                # a single sample, so one floor vibration poisons the whole two
                # seconds; arriving oftener than that, calibration never ends
                # while the operator is in fact holding still. A deviation still
                # catches sustained motion and drift, and rides out one bump.
                # What actually ruins a bias estimate is sustained rotation or
                # a tilt drift, both of which move the mean; symmetric vibration
                # averages out over 200 samples. So these allow roughly three
                # times the datasheet noise, which real wiring reaches, while
                # still rejecting a drift of 0.2 m/s2 per second.
                accel_spread = norm([spread(s[0][i] for s in self.samples) for i in range(3)])
                gyro_spread = norm([spread(s[1][i] for s in self.samples) for i in range(3)])
                stable = accel_spread < .09 and gyro_spread < .008
            state['calibration_progress'] = min(.99, len(self.samples)/self.calibration_samples) if stable else 0.
            state['message'] = ('로봇을 완전히 정지시켜 약 2초간 유지하세요 (일정한 회전도 금지)' if plausible else
                                '정지 보정 범위 초과 — 움직임·배선·센서 상태를 확인하세요')
            if plausible and not stable:
                state['message'] = '값이 변하고 있습니다 — 진동·움직임 없이 약 2초간 유지하세요'
            if len(self.samples) < self.calibration_samples or not stable:
                self.state = state
                return state
            avg = [sum(s[0][i] for s in self.samples)/len(self.samples) for i in range(3)]
            self.bias = [sum(s[1][i] for s in self.samples)/len(self.samples) for i in range(3)]
            self.startup_accel_norm = norm(avg)
            self.accel_warning = abs(self.startup_accel_norm-G) > .35
            self.q = from_euler(math.atan2(avg[1], avg[2]),
                                math.atan2(-avg[0], math.hypot(avg[1], avg[2])), 0.)
            if self.last_mag is not None and now-self.last_mag_time < .25:
                field_world = rotate(self.q, self.last_mag)
                if math.hypot(*field_world[:2]) > 5:
                    self.q = multiply(from_euler(0, 0, -math.atan2(field_world[1], field_world[0])), self.q)
                    self.magnetic = True
                    self.last_mag_accepted = now
            dt = 0.
            self.samples.clear()
        corrected = [gyro[i]-self.bias[i] for i in range(3)]
        omega = corrected[:]
        if abs(norm(acc)-self.startup_accel_norm) < .5:
            gravity_body = rotate(conjugate(self.q), [0, 0, 1])
            error = cross([a/norm(acc) for a in acc], gravity_body)
            omega = [omega[i]+1.5*error[i] for i in range(3)]
        dq = multiply(self.q, [0, *omega])
        q = [self.q[i]+.5*dq[i]*dt for i in range(4)]
        self.q = [x/norm(q) for x in q]
        if self.magnetic and calibrated_mag is not None:
            field_world = rotate(self.q, calibrated_mag)
            error = math.atan2(field_world[1], field_world[0])
            # Reject large heading innovations; smaller magnetic interference is still possible.
            if math.hypot(*field_world[:2]) > 5 and abs(error) < math.radians(45):
                self.q = multiply(from_euler(0, 0, -error*min(1., dt)), self.q)
                self.last_mag_accepted = now
        world_acc = rotate(self.q, acc)
        # Use the stationary startup magnitude as this sensor's local gravity
        # estimate. It removes the immediately observed scalar offset, but is
        # not a replacement for a multi-orientation accelerometer calibration.
        linear = [world_acc[0], world_acc[1], world_acc[2]-self.startup_accel_norm]
        if norm(corrected) < .03 and norm(linear) < .25:
            self.stationary_samples += 1
        else:
            self.stationary_samples = 0
        stationary = self.stationary_samples >= 50  # about 0.5 s at 100 Hz
        if stationary:
            linear = [0., 0., 0.]
        old_velocity = self.velocity[:]
        self.velocity = [old_velocity[i]+.5*(linear[i]+self.previous_linear[i])*dt for i in range(3)]
        self.position = [self.position[i]+.5*(old_velocity[i]+self.velocity[i])*dt for i in range(3)]
        if stationary:
            # Zero-velocity update for a ground robot. IMU alone cannot tell a
            # smooth constant-speed run from rest, so encoders remain required.
            self.velocity = [0., 0., 0.]
        self.previous_linear = linear
        forward = rotate(self.q, [1, 0, 0])
        heading = (-math.degrees(math.atan2(forward[1], forward[0]))) % 360 if math.hypot(*forward[:2]) > .1 else None
        attitude = [math.degrees(value) for value in to_euler(self.q)]
        state.update(status='ready', calibration_progress=1., calibration_skipped=self.skip_calibration,
                     quaternion_wxyz=self.q[:],
                     gyro_bias_rps=self.bias[:], startup_accel_norm_mps2=self.startup_accel_norm,
                     acceleration_calibration_warning=self.accel_warning,
                     gyro_rps=corrected, linear_acceleration_body=rotate(conjugate(self.q), linear),
                     linear_acceleration_world=linear, velocity_world=self.velocity[:],
                     position_world=self.position[:], speed_mps=norm(self.velocity),
                     stationary_detected=stationary,
                     attitude_deg={'roll': attitude[0], 'pitch': attitude[1], 'yaw_ccw': attitude[2]},
                     heading_deg=heading, direction8=direction8(heading, self.magnetic) if heading is not None else None,
                     heading_reference='magnetic' if self.magnetic else 'relative',
                     frame='magnetic_north_west_up' if self.magnetic else 'initial_forward_left_up',
                     magnetic_healthy=self.magnetic and now-self.last_mag_accepted < 1.,
                     message=('자기 북쪽 기준 (자기장 간섭 주의)' if self.magnetic else '시작 방향 기준') +
                             ' · 속도/이동거리는 드리프트하는 IMU 적분 추정값')
        if self.accel_warning:
            state['message'] += ' · 정지 가속도 크기 이상: 가속도·속도·변위 신뢰 낮음, 별도 센서 보정 필요'
        if stationary:
            state['message'] += ' · 정지 감지: 속도 0 보정 적용'
        if gap:
            state['message'] += ' · 수신 공백 후 재개: 속도 적분만 초기화 (정지 보정은 유지)'
        if self.skip_calibration:
            state['message'] += (' · 정지 보정 생략 모드: 자이로 바이어스 미보정 —'
                                 ' 방위는 계속 돌아가고 기울기도 몇 도 오차, 측정값 아님')
        self.state = state
        return state

    def snapshot(self, now=None):
        now = time.time() if now is None else now
        if now-self.state.get('timestamp_unix', 0) > 1.5:
            return {'schema': 1, 'status': 'disconnected', 'message': 'IMU 데이터 수신 중단'}
        return dict(self.state)
