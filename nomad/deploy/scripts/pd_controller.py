#!/usr/bin/env python3

import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from ros_data import ROSData
from utils import clip_angle


class PDController(Node):
    def __init__(self):
        super().__init__('pd_controller')

        self.v_max = self.declare_parameter("v_max", 0.2).value
        self.w_max = self.declare_parameter("w_max", 0.2).value
        self.hz = self.declare_parameter("frame_rate", 4).value
        self.eps = self.declare_parameter("eps", 1e-8).value
        self.flip_ang_vel = self.declare_parameter(
            "flip_ang_vel", np.pi / 4).value
        self.waypoint_timeout = self.declare_parameter(
            "waypoint_timeout", 0.2).value

        self.waypoint = ROSData(self, self.waypoint_timeout, name="waypoint")

        self.reached_goal = False
        self.reverse_mode = False

        self.init_comms()

    def init_comms(self):
        self.waypoint_sub = self.create_subscription(
            Float32MultiArray,
            '/waypoint',
            self.callback_drive,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.reached_goal_sub = self.create_subscription(
            Bool,
            '/reached_goal',
            self.callback_reached_goal,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.vel_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.create_timer(
            1.0 / self.hz,
            self.run_loop,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.get_logger().info("Initialized communication")

    def callback_drive(self, waypoint_msg: Float32MultiArray):
        self.get_logger().info("Setting waypoint")
        self.waypoint.set(waypoint_msg.data)

    def callback_reached_goal(self, reached_goal_msg: Bool):
        self.reached_goal = reached_goal_msg.data

    def pd_controller(self, waypoint: np.ndarray) -> Tuple[float, float]:
        assert len(waypoint) == 2 or len(
            waypoint) == 4, "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint

        dt = 1 / self.hz
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < self.eps and np.abs(dy) < self.eps:
            v = 0
            w = clip_angle(np.arctan2(hy, hx)) / dt
        elif np.abs(dx) < self.eps:
            v = 0
            w = np.sign(dy) * np.pi / (2 * dt)
        else:
            v = dx / dt
            w = np.arctan(dy / dx) / dt
        v = np.clip(v, 0, self.v_max)
        w = np.clip(w, -self.w_max, self.w_max)
        return v, w

    def run_loop(self):
        vel_msg = Twist()
        if self.reached_goal:
            self.vel_pub.publish(vel_msg)
            self.get_logger().info("Reached goal! Stopping...")
            return
        elif self.waypoint.is_valid(verbose=True):
            v, w = self.pd_controller(self.waypoint.get())
            if self.reverse_mode:
                v *= -1
            vel_msg.linear.x = v
            vel_msg.angular.z = w
            self.get_logger().info(f"Publishing new vel: {v}, {w}")
        self.vel_pub.publish(vel_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PDController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
