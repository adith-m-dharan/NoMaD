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
    deploy/code/deployment/__init__.py
    deploy/code/deployment/navigate.py
    deploy/code/deployment/create_topomap.py
    deploy/code/deployment/utils.py
    deploy/code/deployment/controller.py
    deploy/code/deployment/explore.py
    deploy/code/deployment/search.py
    DESTINATION lib/${PROJECT_NAME})

ament_package()
