#!/usr/bin/env python3

import os
import time
from threading import Thread
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
import yaml
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from utils import msg_to_pil, to_numpy, transform_images, load_model
from vint_train.training.train_utils import get_action


class Explore(Node):
    def __init__(self):
        super().__init__('nomad')
        self.model_name = self.declare_parameter(
            "model_name", "nomad").value
        self.model_weights_path = self.declare_parameter(
            "model_weights_path", "").value
        self.model_config_path = self.declare_parameter(
            "model_config_path", "").value

        self.waypoint = self.declare_parameter("waypoint", 2).value
        self.close_threshold = self.declare_parameter("close_threshold", 3).value
        self.num_samples = self.declare_parameter("num_samples", 8).value

        self.v_max = self.declare_parameter("v_max", 0.2).value
        self.w_max = self.declare_parameter("w_max", 0.4).value
        self.hz = self.declare_parameter("hz", 4.0).value
        self.graph_hz = self.declare_parameter("graph_hz", 0.333).value

        self.n_value = self.declare_parameter("n_value", 0).value
        # # Hardcoded parameters
        # self.model_name = "nomad"
        # self.model_weights_path = "/home/adith/Downloads/nomad_ws/src/nomad/deploy/model_weights/nomad.pth"
        # self.model_config_path = "/home/adith/Downloads/nomad_ws/src/nomad/train/config/model.yaml"
        # self.waypoint = 2
        # self.close_threshold = 3
        # self.num_samples = 8
        # self.v_max = 0.5
        # self.w_max = 0.5
        # self.hz = 4.0
        # self.graph_hz = 0.33
        # self.n_value = 0

        self.load_params()
        self.init_comms()

        self.context_queue = []

    def load_params(self):
        # self.get_logger().info(f"Model config path: {self.model_config_path}")
        assert os.path.isfile(self.model_config_path), \
            f"{self.model_config_path} is not a file. Model config path needs to point to a .yaml file"

        with open(self.model_config_path, "r") as fd:
            model_paths = yaml.safe_load(fd)
        # self.get_logger().info(f"Loaded model paths")

        # Check if the required fields are in the configuration
        # required_keys = ["model_type", "context_size", "num_diffusion_iters"]
        # for key in required_keys:
        #     if key not in model_paths:
        #         raise KeyError(f"Key '{key}' not found in the model configuration!")

        self.model_params = model_paths
        self.context_size = model_paths["context_size"]
        self.num_diffusion_iters = model_paths["num_diffusion_iters"]

        # self.get_logger().info(f"Using model type: {self.model_params['model_type']}")

        # Load model weights
        assert os.path.isfile(self.model_weights_path), \
            f"{self.model_weights_path} is not a file. Model weights path needs to point to a file"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(
            self.model_weights_path,
            self.model_params,
            device=self.device
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Initialize the noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def init_comms(self):
        # Initialize ROS2 communications
        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.waypoint_pub = self.create_publisher(Float32MultiArray, "/waypoint", 10)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, "/waypoint_topic", 10)
        self.get_logger().info("Initialized communications.")

    def image_callback(self, msg):
        # self.get_logger().info("Image message received")
        img = msg_to_pil(msg)
        if self.context_size is None:
            return

        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(img)
            # self.get_logger().info(f"Context queue length: {len(self.context_queue)}")
        else:
            self.context_queue.pop(0)
            self.context_queue.append(img)

    def exploration_loop(self):
        self.get_logger().info("Starting exploration loop...")
        rate = self.create_rate(self.hz)
        while rclpy.ok():
            # self.get_logger().info("Exploration loop tick")
            waypoint_msg = Float32MultiArray()
            if len(self.context_queue) > self.context_size:
                # self.get_logger().info("Context queue ready for processing")
                obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
                obs_images = obs_images.to(self.device)
                fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
                mask = torch.ones(1).long().to(self.device)

                with torch.no_grad():
                    obs_cond = self.model(
                        'vision_encoder',
                        obs_img=obs_images,
                        goal_img=fake_goal,
                        input_goal_mask=mask
                    )

                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(self.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(self.num_samples, 1, 1)

                    noisy_action = torch.randn(
                        (self.num_samples, self.model_params["len_traj_pred"], 2), device=self.device
                    )
                    naction = noisy_action

                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
                    for k in self.noise_scheduler.timesteps:
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten())).tolist()
                self.sampled_actions_pub.publish(sampled_actions_msg)

                # Choose and publish waypoint from /waypoint_topic
                naction = naction[self.n_value]
                chosen_waypoint = naction[self.waypoint]

                if self.model_params["normalize"]:
                    chosen_waypoint *= (self.v_max / self.hz)
                waypoint_msg.data = chosen_waypoint.tolist()
                self.get_logger().info(f"Publishing waypoint: {list(waypoint_msg.data)}")
                self.waypoint_pub.publish(waypoint_msg)
            # else:
            #     self.get_logger().info(f"Context queue not ready (current size: {len(self.context_queue)}, required: {self.context_size + 1})")
            rate.sleep()


def main(args=None):
    rclpy.init(args=args)
    node = Explore()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    exploration_thread = Thread(target=node.exploration_loop)
    exploration_thread.start()

    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        exploration_thread.join()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
