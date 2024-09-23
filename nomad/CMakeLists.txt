cmake_minimum_required(VERSION 3.8)
project(nomad)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

install(PROGRAMS
    deploy/scripts/__init__.py
    deploy/scripts/navigate.py
    deploy/scripts/create_topomap.py
    deploy/scripts/utils.py
    deploy/scripts/pd_controller.py
    deploy/scripts/ros_data.py
    DESTINATION lib/${PROJECT_NAME})

ament_package()
