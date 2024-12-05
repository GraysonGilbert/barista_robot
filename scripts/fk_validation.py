#!/usr/bin/env python3

import rclpy # type: ignore
from rclpy.node import Node # type: ignore
from std_msgs.msg import Float64MultiArray # type: ignore
from sympy import symbols, sin, cos, Matrix, Derivative, pi, simplify, init_printing
import math
import numpy as np


class JointMoverNode(Node):
    def __init__(self):
        super().__init__('joint_mover_node')
        
        # Creating Publishers
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)

        init_printing(use_unicode=True)

        # Initializing Symbols
        a_1, a_2, a_3, a_4, a_5, a_6, =  symbols('a_1 a_2 a_3 a_4 a_5 a_6')
        alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6 = symbols('alpha_1 alpha_2 alpha_3 alpha_4 alpha_5 alpha_6')
        d_1, d_2, d_3, d_4, d_5, d_6 = symbols('d_1 d_2 d_3 d_4 d_5 d_6')
        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')

        # Initialize DH Parameters
        a_1 = 0
        a_2 = -.6127
        a_3 = -.57155
        a_4 = 0
        a_5 = 0
        a_6 = 0

        alpha_1 = pi/2
        alpha_2 = 0
        alpha_3 = 0
        alpha_4 = pi/2
        alpha_5 = pi/2
        alpha_6 = 0

        d_1 = .1807
        d_2 = 0
        d_3 = 0
        d_4 = 0.17415
        d_5 = .11985
        d_6 = .11655


        # Constructing Matrices

        A_1 = Matrix([[cos(theta_1 - pi), -cos(alpha_1)*sin(theta_1 - pi), sin(alpha_1)*sin(theta_1 - pi), a_1*cos(theta_1 - pi)],
             [sin(theta_1 - pi), cos(alpha_1)*cos(theta_1 - pi), -sin(alpha_1)*cos(theta_1 - pi), a_1*sin(theta_1 - pi)],
             [0, sin(alpha_1), cos(alpha_1), d_1],
             [0, 0, 0, 1]])

        A_2 = Matrix([[cos(theta_2 - (pi/2)), -cos(alpha_2)*sin(theta_2 - (pi/2)), sin(alpha_2)*sin(theta_2 - (pi/2)), a_2*cos(theta_2 - (pi/2))],
             [sin(theta_2 - (pi/2)), cos(alpha_2)*cos(theta_2 - (pi/2)), -sin(alpha_2)*cos(theta_2 - (pi/2)), a_2*sin(theta_2 - (pi/2))],
             [0, sin(alpha_2), cos(alpha_2), d_2],
             [0, 0, 0, 1]])
        A_3 = Matrix([[cos(theta_3), -cos(alpha_3)*sin(theta_3), sin(alpha_3)*sin(theta_3), a_3*cos(theta_3)],
             [sin(theta_3), cos(alpha_3)*cos(theta_3), -sin(alpha_3)*cos(theta_3), a_3*sin(theta_3)],
             [0, sin(alpha_3), cos(alpha_3), d_3],
             [0, 0, 0, 1]])

        A_4 = Matrix([[cos(theta_4 - (pi/2)), -cos(alpha_4)*sin(theta_4 - (pi/2)), sin(alpha_4)*sin(theta_4 - (pi/2)), a_4*cos(theta_4 - (pi/2))],
             [sin(theta_4 - (pi/2)), cos(alpha_4)*cos(theta_4 - (pi/2)), -sin(alpha_4)*cos(theta_4 - (pi/2)), a_4*sin(theta_4 - (pi/2))],
             [0, sin(alpha_4), cos(alpha_4), d_4],
             [0, 0, 0, 1]])

        A_5 = Matrix([[cos(theta_5), -cos(alpha_5)*sin(theta_5), sin(alpha_5)*sin(theta_5), a_5*cos(theta_5)],
             [sin(theta_5), cos(alpha_5)*cos(theta_5), -sin(alpha_5)*cos(theta_5), a_5*sin(theta_5)],
             [0, sin(alpha_5), cos(alpha_5), d_5],
             [0, 0, 0, 1]])

        A_6 = Matrix([[cos(theta_6), -cos(alpha_6)*sin(theta_6), sin(alpha_6)*sin(theta_6), a_6*cos(theta_6)],
             [sin(theta_6), cos(alpha_6)*cos(theta_6), -sin(alpha_6)*cos(theta_6), a_6*sin(theta_6)],
             [0, sin(alpha_6), cos(alpha_6), d_6],
             [0, 0, 0, 1]])

        # Calculating Transformations
        A_1_wrt_0 = A_1

        A_2_wrt_0 = A_1_wrt_0 * A_2

        A_3_wrt_0 = A_2_wrt_0 * A_3

        A_4_wrt_0 = A_3_wrt_0 * A_4

        A_5_wrt_0 = A_4_wrt_0 * A_5

        self.A_6_wrt_0 = simplify(A_5_wrt_0 * A_6)




        self.timer_period = 1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)



        self.theta_1_vals = 0.0
        self.theta_2_vals = math.pi/2
        self.theta_3_vals = 0.0
        self.theta_4_vals = -math.pi/2
        self.theta_5_vals = -math.pi/2
        self.theta_6_vals = 0.0
    
    def timer_callback(self):
    
        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6')

        j1_theta = float(self.theta_1_vals)
        j2_theta = float(self.theta_2_vals)
        j3_theta = float(self.theta_3_vals)
        j4_theta = float(self.theta_4_vals)
        j5_theta = float(self.theta_5_vals)
        j6_theta = float(self.theta_6_vals)

        self.end_pos = self.A_6_wrt_0.subs({theta_1: j1_theta, theta_2: j2_theta, theta_3: j3_theta, theta_4: j4_theta, theta_5: j5_theta, theta_6: j6_theta})
        self.get_logger().info(f'End Effector Position: {self.end_pos[0,3], self.end_pos[1, 3], self.end_pos[2, 3]}')

        self.j_angle = Float64MultiArray()
        self.j_angle.data = [j1_theta, j2_theta, j3_theta, j4_theta, j5_theta, j6_theta]
  
        self.joint_position_pub.publish(self.j_angle)


def main(args=None):
    rclpy.init(args=args)
    node = JointMoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

