#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math


class JointVelocityNode(Node):
    def __init__(self):
        super().__init__('joint_velocity_node')
        self.get_logger().info("Joint Velocity Node has started.")

        # Robot arm parameters (DH parameters)
        self.a = [0, -0.6127, -0.57155, 0, 0, 0]  # Link lengths
        self.d = [0.1807, 0, 0, 0.17415, 0.11985, 0.11655]  # Link offsets

        # Create a publisher for joint positions
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)

        # Target position (end-effector position in 3D space)
        self.target_position = [1.0,0,0.6]  # Example (x, y, z)

        # Timer to compute and publish joint positions
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(f"Target position set to: {self.target_position}")

    def inverse_kinematics(self, target_position):
        """
        Manually calculate the inverse kinematics for a 6-DOF manipulator.
        """
        try:
            x, y, z = target_position

            # Extract robot parameters
            d1 = self.d[0]  # Base height
            a2 = self.a[1]  # Length of the second link
            a3 = self.a[2]  # Length of the third link
            d4 = self.d[3]  # Offset to the wrist

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

            # Solve for theta_4, theta_5, theta_6 (orientation)
            # Assuming no orientation change (simplified case)
            theta_4 = 0
            theta_5 = 0
            theta_6 = 0

            # Combine all joint angles
            joint_angles = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
            return joint_angles

        except Exception as e:
            self.get_logger().error(f"Error in manual inverse kinematics: {str(e)}")
            return None

    def timer_callback(self):
        self.get_logger().info(f"Attempting to reach target position: {self.target_position}")

        # Solve inverse kinematics
        joint_angles = self.inverse_kinematics(self.target_position)

        if joint_angles:
            self.get_logger().info(f"Computed joint angles: {joint_angles}")

            # Ensure joint_angles is a proper list of floats
            joint_angles = [float(angle) for angle in joint_angles]

            # Publish the joint angles
            msg = Float64MultiArray()
            msg.data = joint_angles
            self.joint_position_pub.publish(msg)

            self.get_logger().info(f"Published joint angles: {joint_angles}")

            # Stop the timer as the goal has been reached
            self.timer.cancel()
            self.get_logger().info("Motion complete. Stopping timer.")
        else:
            self.get_logger().error("Failed to compute inverse kinematics.")


def main(args=None):
    rclpy.init(args=args)
    velocity_node = JointVelocityNode()
    rclpy.spin(velocity_node)
    velocity_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
