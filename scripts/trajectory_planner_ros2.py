#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import numpy as np


class JointVelocityNode(Node):
    def __init__(self):
        super().__init__('joint_velocity_node')
        self.get_logger().info("Joint Velocity Node has started.")

        # Robot arm parameters (DH parameters)
        self.a = [0, -0.6127, -0.57155, 0, 0, 0]  # Link lengths
        self.d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]  # Link offsets

        # Publisher for joint positions
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)

        # Target position (end-effector position in 3D space) and orientation (roll, pitch, yaw)
        self.target_position = [0.4, 0.2, 0.6]  # (x, y, z)
        self.target_orientation = [0.0, 0.0, 0.0]  # (roll, pitch, yaw)

        # Number of steps for smooth trajectory
        self.trajectory_steps = 100
        self.current_step = 0
        self.trajectory = []

        # Timer for executing the trajectory
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(f"Target position set to: {self.target_position}")

        # Compute the trajectory
        self.compute_trajectory()

    def compute_trajectory(self):
        """
        Compute a smooth trajectory for the robot arm using linear interpolation.
        """
        joint_angles_target = self.inverse_kinematics(self.target_position, self.target_orientation)
        if not joint_angles_target:
            self.get_logger().error("Failed to compute inverse kinematics for target position.")
            self.timer.cancel()
            return

        # Assume current joint angles are all zero
        joint_angles_current = [0.0] * 6

        # Linearly interpolate between current and target joint angles
        for step in range(self.trajectory_steps + 1):
            fraction = step / self.trajectory_steps
            interpolated_angles = [
                joint_angles_current[i] + fraction * (joint_angles_target[i] - joint_angles_current[i])
                for i in range(6)
            ]
            self.trajectory.append(interpolated_angles)

        self.get_logger().info("Trajectory computed successfully.")

    def timer_callback(self):
        """
        Publish the next set of joint angles in the trajectory.
        """
        if self.current_step < len(self.trajectory):
            joint_angles = self.trajectory[self.current_step]
            self.publish_joint_angles(joint_angles)
            self.current_step += 1
        else:
            self.get_logger().info("Trajectory execution complete. Stopping timer.")
            self.timer.cancel()

    def inverse_kinematics(self, target_position, target_orientation):
        """
        Calculate the inverse kinematics for a 6-DOF manipulator.
        """
        try:
            x, y, z = target_position
            roll, pitch, yaw = target_orientation

            # Extract robot parameters
            d1 = self.d[0]
            a2 = self.a[1]
            a3 = self.a[2]
            d4 = self.d[3]

            # Solve for theta_1
            theta_1 = math.atan2(y, x)

            # Compute the wrist center position
            wx = x - d4 * math.cos(theta_1)
            wy = y - d4 * math.sin(theta_1)
            wz = z - d1

            # Distance from base to wrist center in the x-y plane
            r = math.sqrt(wx**2 + wy**2)

            # Solve for theta_3 using the law of cosines
            D = (r**2 + wz**2 - a2**2 - a3**2) / (2 * a2 * a3)
            if abs(D) > 1.0:
                raise ValueError("Target position is out of reach.")
            theta_3 = math.atan2(-math.sqrt(1 - D**2), D)

            # Solve for theta_2
            phi_2 = math.atan2(wz, r)
            phi_1 = math.atan2(a3 * math.sin(theta_3), a2 + a3 * math.cos(theta_3))
            theta_2 = phi_2 - phi_1

            # Compute the rotation matrix for the desired orientation
            R_desired = self.rotation_matrix_from_rpy(roll, pitch, yaw)

            # Compute the rotation matrix for the first three joints (R_0_3)
            R_0_3 = self.compute_r_0_3(theta_1, theta_2, theta_3)

            # Compute R_4_6
            R_4_6 = np.linalg.inv(R_0_3).dot(R_desired)

            # Extract theta_4, theta_5, theta_6 from R_4_6
            theta_5 = math.atan2(math.sqrt(R_4_6[0, 2]**2 + R_4_6[1, 2]**2), R_4_6[2, 2])
            theta_4 = math.atan2(R_4_6[1, 2], R_4_6[0, 2])
            theta_6 = math.atan2(R_4_6[2, 1], -R_4_6[2, 0])

            # Combine all joint angles
            return [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]

        except Exception as e:
            self.get_logger().error(f"Error in inverse kinematics: {str(e)}")
            return None

    def rotation_matrix_from_rpy(self, roll, pitch, yaw):
        """
        Compute a rotation matrix from roll, pitch, and yaw.
        """
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        return R_z.dot(R_y).dot(R_x)

    def compute_r_0_3(self, theta_1, theta_2, theta_3):
        """
        Compute the rotation matrix R_0_3 using theta_1, theta_2, and theta_3.
        """
        R_z1 = np.array([
            [math.cos(theta_1), -math.sin(theta_1), 0],
            [math.sin(theta_1), math.cos(theta_1), 0],
            [0, 0, 1]
        ])
        R_y2 = np.array([
            [math.cos(theta_2), 0, math.sin(theta_2)],
            [0, 1, 0],
            [-math.sin(theta_2), 0, math.cos(theta_2)]
        ])
        R_y3 = np.array([
            [math.cos(theta_3), 0, math.sin(theta_3)],
            [0, 1, 0],
            [-math.sin(theta_3), 0, math.cos(theta_3)]
        ])
        return R_z1.dot(R_y2).dot(R_y3)

    def publish_joint_angles(self, joint_angles):
        """
        Publish joint angles to the position controller.
        """
        msg = Float64MultiArray()
        msg.data = joint_angles
        self.joint_position_pub.publish(msg)
        self.get_logger().info(f"Published joint angles: {joint_angles}")


def main(args=None):
    rclpy.init(args=args)
    node = JointVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
