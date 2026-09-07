/* Independent of camera polling. Never show stale readings as live measurements. */
(() => {
    'use strict';
    const get = id => document.getElementById(`imu-${id}`);
    if (!get('status')) return;
    const relative = ['기준', '우전', '우', '우후', '반대', '좌후', '좌', '좌전'];
    const magnetic = ['북', '북동', '동', '남동', '남', '남서', '서', '북서'];
    const number = v => typeof v === 'number' && Number.isFinite(v);
    const vector = v => Array.isArray(v) && v.length === 3 && v.every(number);
    const quaternion = q => Array.isArray(q) && q.length === 4 && q.every(number) && Math.abs(Math.hypot(...q) - 1) < .1;
    const xyz = (v, unit) => vector(v) ? `${v.map(x => x.toFixed(3)).join(' / ')} ${unit}` : '—';
    function modelMatrix(q) {
        const [w, x, y, z] = q;
        const r = [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ];
        const c = [[0,-1,0],[-1,0,0],[0,0,1]];
        const multiply3 = (a, b) => a.map(row => b[0].map((_, j) => row.reduce((sum, value, k) => sum + value*b[k][j], 0)));
        const transpose = a => a[0].map((_, i) => a.map(row => row[i]));
        const m = multiply3(multiply3(c, r), transpose(c));
        return `matrix3d(${m[0][0]},${m[1][0]},${m[2][0]},0,${m[0][1]},${m[1][1]},${m[2][1]},0,${m[0][2]},${m[1][2]},${m[2][2]},0,0,0,0,1)`;
    }
    function render(data) {
        const ready = data.status === 'ready';
        document.querySelector('.imu-panel').dataset.disconnected = String(!ready);
        const north = data.heading_reference === 'magnetic';
        const status = get('status');
        status.dataset.state = data.status;
        status.dataset.demo = String(Boolean(data.demo));
        // Calibrating is a live state, not a dead one. Showing the progress
        // and the raw reading is what tells the operator the board is talking.
        const percent = number(data.calibration_progress) ? ` ${Math.round(data.calibration_progress*100)}%` : '';
        status.textContent = `${data.demo ? '데모 · ' : ''}${ready ? (data.stationary_detected ? '수신 중 · 정지' : '수신 중') : data.status === 'calibrating' ? `정지 보정 중${percent}` : '연결 끊김'}`;
        get('sensor').textContent = data.chip ? `${data.chip} / ${data.mag_present ? '감지됨' : '없음'}` : '—';
        document.querySelectorAll('[data-imu-dir]').forEach(el => {
            el.textContent = (north ? magnetic : relative)[Number(el.dataset.imuDir)];
        });
        get('reference').textContent = north ? '자기 북쪽 기준 · 진북 아님' : '시작 방향 기준 · 로봇 전방';
        const headingValid = ready && number(data.heading_deg);
        get('arrow').setAttribute('visibility', headingValid ? 'visible' : 'hidden');
        if (headingValid) get('arrow').setAttribute('transform', `rotate(${data.heading_deg} 120 120)`);
        get('heading').textContent = headingValid ? `${data.direction8} · ${data.heading_deg.toFixed(1)}°` : '—';
        get('compass-title').textContent = `로봇 전방 방향: ${get('heading').textContent}`;
        const attitudeValid = ready && quaternion(data.quaternion_wxyz);
        if (attitudeValid) get('model').style.transform = modelMatrix(data.quaternion_wxyz);
        const attitude = data.attitude_deg;
        get('attitude-title').textContent = attitudeValid && attitude && number(attitude.roll) && number(attitude.pitch) && number(attitude.yaw_ccw)
            ? `Roll ${attitude.roll.toFixed(1)}° / Pitch ${attitude.pitch.toFixed(1)}° / Yaw ${attitude.yaw_ccw.toFixed(1)}°`
            : 'Roll / Pitch / Yaw — 데이터 대기';
        get('speed').textContent = ready && number(data.speed_mps) ? `${data.speed_mps.toFixed(3)} m/s` : '—';
        const acc = data.linear_acceleration_body;
        get('acceleration').textContent = ready && vector(acc) ? `${Math.hypot(...acc).toFixed(3)} m/s²` : '—';
        // Gravity cannot be removed before the bias is known, so during
        // calibration this row carries the raw reading and says so.
        const calibrating = data.status === 'calibrating';
        get('acceleration-xyz-label').textContent = ready || !calibrating
            ? '가속도 X / Y / Z (로봇 좌표)'
            : '가속도 X / Y / Z (원시 · 중력 포함)';
        get('acceleration-xyz').textContent = ready ? xyz(acc, 'm/s²')
            : calibrating ? xyz(data.accel_mps2, 'm/s²') : '—';
        get('position').textContent = ready ? xyz(data.position_world, 'm') : '—';
        get('message').textContent = (data.demo ? '모의 데이터 — 실제 측정 아님. ' : '') +
            (data.message || 'IMU 데이터 대기') +
            (ready && north && !data.magnetic_healthy ? ' · 자력계 보정 중단: 자이로로 방향 유지 중, 간섭/연결 확인' : '');
    }
    async function poll() {
        if (!document.hidden) {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 1200);
            try {
                const response = await fetch('/api/imu', { cache: 'no-store', signal: controller.signal });
                if (!response.ok) throw new Error('IMU API unavailable');
                const data = await response.json();
                if (!data || data.schema !== 1) throw new Error('Invalid IMU data');
                if (data.status !== 'disconnected' && (!number(data.timestamp_unix) || Math.abs(Date.now()/1000-data.timestamp_unix) > 2)) {
                    throw new Error('Stale IMU sample');
                }
                render(data);
            } catch (_) {
                render({ status: 'disconnected', message: 'IMU 브리지 미연결 — bridge.py 실행 필요' });
            } finally { clearTimeout(timer); }
        }
        setTimeout(poll, 300);
    }
    poll();
})();
