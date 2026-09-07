import copy
import importlib.util
import math
from pathlib import Path
import unittest

from calibrate_mag import fit
from imu_core import G, Processor, conjugate, direction8, from_euler, norm, rotate, to_euler


def packet(i, acc=None, gyro=None, mag=None):
    return {'v': 1, 'type': 'imu', 'seq': i & 0xffffffff, 't_us': (i*10000) & 0xffffffff,
            'chip': 'MPU9250', 'mag_present': True, 'accel_mps2': acc or [0, 0, G],
            'gyro_rps': gyro or [0, 0, 0], 'mag_uT': mag}


class ImuTests(unittest.TestCase):
    def calibrated(self, **kwargs):
        p = Processor(calibration_samples=5, **kwargs)
        for i in range(5):
            p.update(packet(i), now=100+i*.01)
        return p

    def test_stationary_gravity_removed(self):
        p = self.calibrated()
        for i in range(5, 205):
            s = p.update(packet(i), now=100+i*.01)
        self.assertAlmostEqual(s['speed_mps'], 0, places=8)
        self.assertLess(norm(s['linear_acceleration_body']), 1e-8)
        self.assertEqual(s['direction8'], '기준')

    def test_clockwise_quarter_turn(self):
        p = self.calibrated()
        for i in range(5, 105):
            s = p.update(packet(i, gyro=[0, 0, -math.pi/2]))
        self.assertAlmostEqual(s['heading_deg'], 90, places=1)
        self.assertEqual(s['direction8'], '우')
        self.assertAlmostEqual(norm(s['quaternion_wxyz']), 1)

    def test_tilt_gravity_removal(self):
        p = Processor(calibration_samples=5)
        q = from_euler(.4, -.3, 0)
        acc = rotate(conjugate(q), [0, 0, G])
        for i in range(100):
            s = p.update(packet(i, acc=acc))
        self.assertLess(norm(s['linear_acceleration_body']), 1e-8)

    def test_linear_acceleration_integrates(self):
        p = self.calibrated()
        for i in range(5, 105):
            s = p.update(packet(i, acc=[0, 0, G+1]))
        self.assertAlmostEqual(s['velocity_world'][2], .995, places=6)
        self.assertAlmostEqual(s['position_world'][2], .495025, places=5)

    def test_stationary_zero_velocity_update(self):
        p = self.calibrated()
        p.velocity = [1., 0., 0.]
        for i in range(5, 55):
            s = p.update(packet(i))
        self.assertAlmostEqual(s['speed_mps'], 0.)
        self.assertTrue(s['stationary_detected'])

    def test_motion_does_not_zero_velocity(self):
        p = self.calibrated()
        p.velocity = [1., 0., 0.]
        for i in range(5, 105):
            s = p.update(packet(i, acc=[1, 0, G]))
        self.assertGreater(s['speed_mps'], 1.)
        self.assertFalse(s['stationary_detected'])

    def test_moving_cannot_calibrate(self):
        p = Processor(calibration_samples=5)
        for i in range(100):
            s = p.update(packet(i, gyro=[0, 0, 1]))
        self.assertEqual(s['status'], 'calibrating')
        self.assertEqual(s['calibration_progress'], 0)

    def test_gap_keeps_calibration_but_breaks_integration(self):
        # A gap in reception is not a reason to make the operator hold still
        # again. The gyro bias measured before it is still this sensor's bias;
        # only dead reckoning across the blackout is unrecoverable.
        p = self.calibrated()
        p.velocity = [1, 2, 3]
        s = p.update(packet(100))
        self.assertEqual(s['status'], 'ready')
        self.assertTrue(s['stream_gap'])
        self.assertEqual(p.velocity, [0, 0, 0])
        self.assertEqual(p.epoch, 1)

    def test_device_reboot_recalibrates(self):
        p = self.calibrated()
        s = p.update(packet(0), now=200)  # sequence and device clock rewound
        self.assertEqual(s['status'], 'calibrating')
        self.assertIsNone(p.bias)
        self.assertEqual(p.epoch, 2)

    def test_observed_stationary_bias_can_calibrate(self):
        p = Processor()
        for i in range(200):
            s = p.update(packet(i, acc=[-.24, -.04, 9.16+.015*math.sin(i)],
                                gyro=[-.119+.001*math.sin(i), .038, .007]))
        self.assertEqual(s['status'], 'ready')
        self.assertTrue(s['acceleration_calibration_warning'])
        self.assertLess(norm(s['gyro_rps']), .003)
        self.assertIsNotNone(s['heading_deg'])
        self.assertLess(abs(s['linear_acceleration_world'][2]), .03)

    def test_variable_rotation_rejected_then_recovers(self):
        p = Processor()
        for i in range(250):
            s = p.update(packet(i, gyro=[.12+.04*math.sin(i), 0, 0]))
        self.assertEqual(s['status'], 'calibrating')
        for i in range(250, 450):
            s = p.update(packet(i, gyro=[.12, 0, 0]))
        self.assertEqual(s['status'], 'ready')

    def test_slow_tilt_during_window_rejected(self):
        p = Processor()
        for i in range(200):
            s = p.update(packet(i, acc=[i*.005, 0, G], gyro=[0, .1, 0]))
        self.assertEqual(s['status'], 'calibrating')

    def test_gross_accel_error_stays_blocked(self):
        p = Processor()
        for i in range(200):
            s = p.update(packet(i, acc=[0, 0, 5]))
        self.assertEqual(s['status'], 'calibrating')

    def test_timer_wrap(self):
        p = self.calibrated()
        p.previous = (0xfffffff0, 10)
        v = packet(11)
        v['t_us'] = (0xfffffff0+10000) & 0xffffffff
        self.assertEqual(p.update(v)['status'], 'ready')
        self.assertEqual(p.epoch, 1)

    def test_bad_sample_does_not_mutate_state(self):
        p = self.calibrated()
        before = copy.deepcopy(p.state)
        v = packet(5)
        v['accel_mps2'][0] = float('nan')
        with self.assertRaises(ValueError):
            p.update(v)
        self.assertEqual(before, p.state)

    def test_stale_hides_values(self):
        p = self.calibrated()
        s = p.snapshot(now=102)
        self.assertEqual(s['status'], 'disconnected')
        self.assertNotIn('speed_mps', s)

    def test_directions_and_boundaries(self):
        self.assertEqual(direction8(359), '기준')
        self.assertEqual(direction8(22.5), '우전')
        self.assertEqual(direction8(270, True), '서')
        self.assertEqual(len({direction8(i*45, True) for i in range(8)}), 8)

    def test_euler_round_trip_and_upside_down(self):
        angles = to_euler(from_euler(.4, -.3, .2))
        for actual, expected in zip(angles, [.4, -.3, .2]):
            self.assertAlmostEqual(actual, expected)
        upside_down = to_euler(from_euler(math.pi, 0, 0))
        self.assertAlmostEqual(abs(upside_down[0]), math.pi)

    def test_mounting(self):
        p = Processor(axes=(2, -1, 3))
        self.assertEqual(p.mounted([1, 2, 3]), [2, -1, 3])
        with self.assertRaises(ValueError):
            Processor(axes=(1, 2, -3))

    def test_magnetic_east_and_rejection(self):
        cal = {'schema': 1, 'axes': [1, 2, 3], 'offset_uT': [0, 0, 0], 'scale': [1, 1, 1], 'field_uT': 40}
        p = Processor(calibration_samples=5, mag_calibration=cal)
        for i in range(5):
            s = p.update(packet(i, mag=[0, 40, 0]), now=100+i*.01)
        self.assertEqual(s['heading_reference'], 'magnetic')
        self.assertAlmostEqual(s['heading_deg'], 90)
        self.assertEqual(s['direction8'], '동')
        for i in range(5, 150):
            s = p.update(packet(i, mag=[400, 0, 0]), now=100+i*.01)
        self.assertFalse(s['magnetic_healthy'])
        self.assertAlmostEqual(s['heading_deg'], 90)

    def test_uncalibrated_mag_does_not_claim_north(self):
        p = Processor(calibration_samples=5)
        for i in range(5):
            s = p.update(packet(i, mag=[0, 40, 0]))
        self.assertEqual(s['heading_reference'], 'relative')

    def test_mag_calibration_coverage(self):
        samples = [[10+30*math.sin(i*.1)*math.cos(i*.3), -5+30*math.sin(i*.1)*math.sin(i*.3), 2+30*math.cos(i*.1)] for i in range(1000)]
        cal = fit(samples, [1, 2, 3])
        self.assertAlmostEqual(cal['offset_uT'][0], 10, delta=.3)
        with self.assertRaises(ValueError):
            fit([[30, 0, 0]]*300, [1, 2, 3])

    def test_ros_magnetic_nwu_to_enu(self):
        path = Path(__file__).parent/'ros2'/'imu_bridge_node.py'
        spec = importlib.util.spec_from_file_location('ros_adapter', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        q, position = module.ros_pose({'quaternion_wxyz': [1, 0, 0, 0], 'position_world': [1, 0, 0], 'heading_reference': 'magnetic'})
        self.assertAlmostEqual(position[0], 0)
        self.assertAlmostEqual(position[1], 1)
        self.assertAlmostEqual(rotate(q, [1, 0, 0])[1], 1)


if __name__ == '__main__':
    unittest.main()
