cmake_minimum_required(VERSION 3.8)
project(barista_robot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(robot_state_publisher REQUIRED)
find_package(xacro REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(ros2_control REQUIRED)


# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)


install(DIRECTORY
urdf
meshes
launch
scripts
worlds
#competition
config
DESTINATION share/${PROJECT_NAME}/
)


install(
  PROGRAMS
  launch/robot_description_publisher.py
  scripts/fk_validation.py
  scripts/ik_validation.py
  scripts/start_position_path.py
  scripts/joint_torque_control.py
  scripts/coffee_path.py
  #scripts/teleop.py
  #scripts/straight_line_controller.py
  #scripts/p_controller.py
  #scripts/laser_scan_relay.py
  DESTINATION lib/${PROJECT_NAME}
  )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
