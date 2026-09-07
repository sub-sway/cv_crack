"""Standalone ROS2 adapter: source ROS2, then python3 ros2/imu_bridge_node.py.

No TF is broadcast. Mounted vectors describe a virtual body-aligned IMU frame.
Magnetic NWU is converted to magnetic ENU; relative heading remains arbitrary.
"""
import json
import math
import sys
import time
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_core import conjugate, from_euler, multiply, rotate


def ros_pose(state):
    q = state['quaternion_wxyz']
    position = state['position_world']
    if state['heading_reference'] == 'magnetic':
        turn = from_euler(0, 0, math.pi/2)
        return multiply(turn, q), rotate(turn, position)
    return q, position


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.duration import Duration
    from sensor_msgs.msg import Imu, MagneticField
    from nav_msgs.msg import Odometry

    class BridgeNode(Node):
        def __init__(self):
            super().__init__('teensy_imu_bridge')
            self.declare_parameter('bridge_url', 'http://127.0.0.1:8766/api/imu')
            self.declare_parameter('frame_id', 'imu_link')
            self.declare_parameter('publish_experimental_odom', False)
            self.declare_parameter('allow_demo', False)
            self.imu = self.create_publisher(Imu, 'imu/data', 10)
            self.mag = self.create_publisher(MagneticField, 'imu/mag_raw', 10)
            self.odom = self.create_publisher(Odometry, 'imu/odometry_experimental', 10)
            self.previous = None
            self.create_timer(.02, self.poll)
            self.get_logger().warning('Prototype fusion; unknown IMU covariance. No TF. Inertial odometry is experimental.')

        def poll(self):
            try:
                with urlopen(self.get_parameter('bridge_url').value, timeout=.2) as response:
                    s = json.load(response)
                if s.get('schema') != 1 or s.get('status') != 'ready':
                    return
                if s.get('demo') and not self.get_parameter('allow_demo').value:
                    return
                age = time.time()-s['timestamp_unix']
                if not math.isfinite(age) or age < -.1 or age > 1.5:
                    return
                key = (s['epoch'], s['seq'], s['timestamp_unix'])
                if key == self.previous:
                    return
                self.previous = key
                q, position = ros_pose(s)
                msg = Imu()
                msg.header.stamp = (self.get_clock().now()-Duration(seconds=max(0., age))).to_msg()
                msg.header.frame_id = self.get_parameter('frame_id').value
                msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = q
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = s['gyro_rps']
                # Specific force includes gravity; level stationary Z should be +9.80665.
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = s['accel_mps2']
                # Zero covariance arrays mean UNKNOWN for sensor_msgs/Imu, not perfect precision.
                self.imu.publish(msg)
                if s.get('mag_uT') is not None:
                    magnetic = MagneticField()
                    magnetic.header = msg.header
                    magnetic.magnetic_field.x, magnetic.magnetic_field.y, magnetic.magnetic_field.z = [v*1e-6 for v in s['mag_uT']]
                    self.mag.publish(magnetic)
                if self.get_parameter('publish_experimental_odom').value:
                    odom = Odometry()
                    odom.header.stamp = msg.header.stamp
                    prefix = 'imu_magnetic_enu' if s['heading_reference'] == 'magnetic' else 'imu_relative'
                    odom.header.frame_id = f'{prefix}_experimental_{s["epoch"]}'
                    odom.child_frame_id = msg.header.frame_id
                    odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = position
                    odom.pose.pose.orientation = msg.orientation
                    velocity_body = rotate(conjugate(s['quaternion_wxyz']), s['velocity_world'])
                    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = velocity_body
                    odom.twist.twist.angular = msg.angular_velocity
                    # Deliberately huge placeholders; not statistically calibrated. Do not feed into navigation.
                    for i in range(6):
                        odom.pose.covariance[i*7] = 1e6
                        odom.twist.covariance[i*7] = 1e6
                    self.odom.publish(odom)
            except (OSError, ValueError, KeyError, TypeError):
                return  # No stale publication when the bridge is unavailable.

    rclpy.init()
    node = BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
