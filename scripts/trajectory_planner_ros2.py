#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sympy import symbols, Matrix, cos, sin, pi, eye
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner_node')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.trajectory_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)

        # Robot configuration
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_joint_angles = np.array([0.0, 0.1, -0.383, 0.283, 0.0001, 0.0])  # Initial angles
        self.trajectory = []

        # Define DH Parameters
        self.dh_params = [
            (0, pi / 2, 0.1833, symbols('theta_1')),
            (-0.73731, 0, 0, symbols('theta_2')),
            (-0.3878, 0, 0, symbols('theta_3')),
            (0, pi / 2, 0.0955, symbols('theta_4')),
            (0, -pi / 2, 0.1155, symbols('theta_5')),
            (0, 0, 0.1218, symbols('theta_6')),
        ]

        # Generate trajectory
        self.generate_trajectory()

        # Timer for publishing
        self.timer = self.create_timer(0.01, self.publish_joint_states)  # 100 Hz

    def dh_matrix(self, a, alpha, d, theta):
        """Returns the DH transformation matrix."""
        return Matrix([
            [cos(theta), -cos(alpha) * sin(theta), sin(alpha) * sin(theta), a * cos(theta)],
            [sin(theta), cos(alpha) * cos(theta), -sin(alpha) * cos(theta), a * sin(theta)],
            [0, sin(alpha), cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward_kinematics(self, joint_angles):
        """Compute the forward kinematics using DH parameters and joint angles."""
        T = eye(4)  # Start with the identity matrix
        for (a, alpha, d, theta), angle in zip(self.dh_params, joint_angles):
            T *= self.dh_matrix(a, alpha, d, theta + angle)
        return T

    def compute_jacobian(self, joint_angles):
        """Compute the Jacobian matrix."""
        T = eye(4)
        Z_vectors = []
        P_vectors = [Matrix([0, 0, 0])]  # Base position
        joint_syms = symbols('theta_1:7')  # Symbolic joint angles

        for i, (a, alpha, d, theta) in enumerate(self.dh_params):
            T *= self.dh_matrix(a, alpha, d, theta + joint_syms[i])
            Z_vectors.append(T[:3, 2])
            P_vectors.append(T[:3, 3])

        P_end = P_vectors[-1]
        J = Matrix.hstack(
            *[Z.cross(P_end - P_vectors[i]) for i, Z in enumerate(Z_vectors)],
            *Z_vectors
        )
        return np.array(J.subs({joint_syms[i]: angle for i, angle in enumerate(joint_angles)})).astype(float)

    def generate_trajectory(self):
        """Generate a trajectory from start to end positions."""
        start_pos = np.array([0.0, 0.0, 0.2])  # Example start position
        end_pos = np.array([0.1, 0.1, 0.4])    # Example end position
        steps = 100

        # Generate linear trajectory
        trajectory_positions = np.linspace(start_pos, end_pos, steps)

        for target_position in trajectory_positions:
            try:
                # Compute the pseudoinverse of the Jacobian
                inv_J = np.linalg.pinv(self.compute_jacobian(self.current_joint_angles))

                # Ensure the target_position matches the Jacobian input
                target_velocity = target_position.reshape(-1, 1)  # Reshape to a column vector (3x1)

                # Perform matrix multiplication
                joint_velocities = inv_J[:3, :] @ target_velocity  # Use only the first 3 rows of the Jacobian for position
                joint_velocities = joint_velocities.flatten()  # Convert back to 1D for updates

                self.current_joint_angles += joint_velocities * 0.01  # Small timestep
                self.trajectory.append(self.current_joint_angles.tolist())
            except np.linalg.LinAlgError as e:
                self.get_logger().error(f"Jacobian computation failed: {str(e)}")
                break

    def publish_joint_states(self):
        """Publish joint states for visualization."""
        if not self.trajectory:
            self.get_logger().info("Trajectory complete!")
            return

        # Create JointState message
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        joint_state_msg.position = self.trajectory.pop(0)
        self.joint_pub.publish(joint_state_msg)

        # Publish trajectory point for compatibility
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_state_msg.position
        point.time_from_start = rclpy.duration.Duration(seconds=0.01).to_msg()
        traj_msg.points.append(point)
        self.trajectory_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlanner()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
