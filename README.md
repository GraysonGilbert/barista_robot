## PROJECT - Autonomous Coffee-Making Barista Robot

This README provides instructions for executing the package, including commands for simulation, operating the robot, and visualization. The project simulates an autonomous barista robot in Gazebo using ROS 2 Galactic, capable of making coffee through automated actions.

For a detailed explanation of the math and process followed to complete this project, please refer to Group_2_final_report.pdf in the repository.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Launching the Simulation](#launching-the-simulation)
- [Coffee-Making Task Execution](#coffee-making-task-execution)
- [Visualization in RViz](#visualization-in-rviz)

## Prerequisites

Ensure the following are installed and configured:

- **Ubuntu 20.04** with **ROS 2 Galactic**
- **Gazebo** simulator
- **Workspace Setup**: The package should be placed in a ROS 2 workspace, built, and sourced.

## Installation

To build the package from GitHub, follow these steps:

1. **Clone the Repository**: Clone the project repository into the `src` directory of your ROS 2 workspace:

   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/GraysonGilbert/barista_robot.git
   ```

2. **Install ROS 2 Dependencies**: Install the required ROS 2 packages to ensure the proper functioning of robotic arm control and perception components:

   ```bash
   sudo apt install ros-galactic-gazebo-ros2-control ros-galactic-ros2-controllers
   ```

3. **Build and Source the Workspace**: Compile the package by building the workspace and source the setup file to overlay this workspace onto your environment:

   ```bash
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
   ```

## Launching the Simulation

To launch the barista robot in the Gazebo environment:

1. **Start the Gazebo Simulation**:

   ```bash
   ros2 launch barista_robot_project gazebo.launch.py
   ```

   This will start the Gazebo environment with the barista robot in the defined workspace (including milk and coffee dispensers, coffee cup and serving tables) , ready to perform coffee-making tasks.

## Coffee-Making Task Execution

1. **Move to Start Position**: To home the robot and move it away from singularities and unreachable workspace positions:

   ```bash
   ros2 run barista_robot start_position_path
   ```

2. **Get Camera Feed**: To start the camera feed for visual input:

   ```bash
   ros2 run barista_robot camera_subscriber
   ```

3. **Execute Coffee-Making Sequence**: To start the robot's coffee-making sequence, run the following command to manage the end-to-end coffee-making task, including picking up a cup, dispensing coffee, adding ingredients, and serving:

   ```bash
   ros2 run barista_robot coffee_path
   ```

## Visualization in RViz

To visualize the robot model and its operations in RViz:

1. **Launch RViz**: Open a new terminal and run the following command to start RViz with the appropriate configuration:

   ```bash
   ros2 launch barista_robot display.launch.py
   ```

2. **Visualize Robot Movements and Environment**:

   - **Robot Model**: Add the "RobotModel" display type and set the topic to `/robot_description` to visualize the barista robot in RViz.
   - **Camera Data**: Add the "Image" display type and set the topic to '/camera/camera_sensor/image_raw' to visualize how the robot is perceiving its environment during the coffee-making task. We also need to change the QoS Reliability Policy from Reliable to Best Effort in the Image display type.

## Additional Information

- **Workspace Setup**: Ensure the workspace is properly sourced every time a new terminal is opened:

  ```bash
  source ~/ros2_ws/install/setup.bash
  ```

- **Simulation Reset**: To reset the simulation or restart the coffee-making process, use the reset command in Gazebo or re-run the launch files.

- **Troubleshooting**: If you encounter issues with the robot's movements or actions, check for errors in the controller script or ensure all required ROS 2 packages are installed and properly configured.

