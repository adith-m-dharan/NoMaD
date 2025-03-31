#!/usr/bin/env python3

import os
import time
import argparse
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
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from utils import Rate
from utils import msg_to_pil, to_numpy, transform_images, load_model
from deploy.code.training.train_utils import get_action


class Search(Node):
    def __init__(self, rotate=False):
        super().__init__('nomad')
        self.rotate = rotate
        
        self.model_name = self.declare_parameter(
            "model_name", "nomad").value
        self.model_weights_path = self.declare_parameter(
            "model_weights_path", "").value
        self.model_config_path = self.declare_parameter(
            "model_config_path", "").value

        self.target_dir = self.declare_parameter(
            "target_dir", "").value

        self.waypoint = self.declare_parameter("waypoint", 2).value
        self.goal_node = self.declare_parameter("goal_node", -1).value
        self.close_threshold = self.declare_parameter("close_threshold", 3).value
        self.radius = self.declare_parameter("radius", 4).value
        self.num_samples = self.declare_parameter("num_samples", 8).value

        self.v_max = self.declare_parameter("v_max", 0.2).value
        self.w_max = self.declare_parameter("w_max", 0.4).value
        self.hz = self.declare_parameter("hz", 4.0).value
        self.graph_hz = self.declare_parameter("graph_hz", 0.333).value

        self.n_value = self.declare_parameter("n_value", 0).value
        self.target_confidence = self.declare_parameter("t_conf", 3).value

        self.load_params()
        self.load_target()
        self.init_comms()

        self.context_queue = []
        self.subgoal = []

    def load_target(self):
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)
        assert self.target_dir != "", "Path to target dir cannot be empty"
        topomap_filenames = sorted(
            os.listdir(self.target_dir),
            key=lambda x: int(x.split(".")[0])
        )
        num_nodes = len(os.listdir(self.target_dir))

        self.target = []
        for i in range(num_nodes):
            img_path = os.path.join(self.target_dir, topomap_filenames[i])
            with PILImage.open(img_path) as img:
                img.load()
                self.target.append(img)

        self.get_logger().info(f"Loaded {len(self.target)} imgs into target")

        self.reached = False

    def load_params(self):
        assert os.path.isfile(self.model_config_path), \
            f"{self.model_config_path} is not a file. Model config path needs to point to a .yaml file"
        with open(self.model_config_path) as fd:
            self.model_params = yaml.safe_load(fd)

        self.context_size = self.model_params["context_size"]
        assert os.path.isfile(self.model_weights_path), \
            f"{self.model_weights_path} is not a file. Last checkpoint path needs to be a file"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(
            self.model_weights_path,
            self.model_params,
            device=self.device
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.model_params["model_type"] == "nomad":
            self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params["num_diffusion_iters"],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon'
            )

    def init_comms(self):
        self.img_sub = self.create_subscription(
            Image,
            "/img",
            self.img_sub_cb,
            10,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.waypoint_pub = self.create_publisher(
            Float32MultiArray,
            "/waypoint",
            10
        )

        self.vel_pub = self.create_publisher(
            Twist,
            "/vel",
            10
        )

        self.sampled_actions_pub = self.create_publisher(
            Float32MultiArray,
            "/sampled_actions",
            10
        )

        self.goal_reached_pub = self.create_publisher(
            Bool,
            "/topoplan/reached_goal",
            10
        )

        self.get_logger().info("Init comms done.")

    def img_sub_cb(self, msg):
        img = msg_to_pil(msg)
        if self.context_size is None:
            return

        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(img)
            return

        self.context_queue.pop(0)
        self.context_queue.append(img)

    def search_loop(self):
        rate = Rate(hz=self.hz)
        consecutive_true_count = 0

        while rclpy.ok():
            chosen_waypoint = np.zeros(4)
            if len(self.context_queue) > self.model_params["context_size"]:
                imgs = transform_images(
                    self.context_queue,
                    self.model_params["image_size"],
                    center_crop=False
                )
                imgs = torch.split(imgs, 3, dim=1)
                imgs = torch.cat(imgs, dim=1)
                imgs = imgs.to(device=self.device)

                mask = torch.zeros(1).long().to(self.device)

                target_image = [self.target[0]]

                goal_img = transform_images(
                    target_image,
                    self.model_params["image_size"],
                    center_crop=False
                ).to(self.device)

                obsgoal_cond = self.model(
                    'vision_encoder',
                    obs_img=imgs.repeat(len(goal_img), 1, 1, 1),
                    goal_img=goal_img,
                    input_goal_mask=mask.repeat(len(goal_img))
                )

                dists = self.model(
                    "dist_pred_net",
                    obsgoal_cond=obsgoal_cond
                )

                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)

                self.get_logger().info(f'distance = {dists[min_idx]}')

                condition_true = dists[min_idx] < self.close_threshold * 2
                if condition_true:
                    consecutive_true_count += 1
                    print(f'count: {consecutive_true_count}')
                else:
                    consecutive_true_count = 0
                if consecutive_true_count >= self.target_confidence:
                    print(f'confidence: {self.target_confidence}')
                    self.reached = True
                sg_idx = min(min_idx + int(condition_true), len(obsgoal_cond) - 1)
                obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)

                if self.rotate:
                    vel_msg = Twist()
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = self.w_max / 2
                    self.vel_pub.publish(vel_msg)
                    rate.sleep()

                with torch.no_grad():
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(self.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(self.num_samples, 1, 1)

                    noisy_action = torch.randn(
                        (self.num_samples,
                        self.model_params["len_traj_pred"], 2),
                        device=self.device
                    )
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(
                        self.num_diffusion_iters)

                    for k in self.noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )
                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample

                naction = to_numpy(get_action(naction))
                sampled_actions_msg = Float32MultiArray()
                sampled_actions_msg.data = np.concatenate(
                    (np.array([0]), naction.flatten())).tolist()

                if rclpy.ok():
                    self.sampled_actions_pub.publish(sampled_actions_msg)
                naction = naction[self.n_value]
                chosen_waypoint = naction[self.waypoint]

            if self.model_params["normalize"]:
                chosen_waypoint[:2] *= (self.v_max/self.hz)
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint.tolist()

            if rclpy.ok():
                self.waypoint_pub.publish(waypoint_msg)

            msg = Bool()
            msg.data = bool(self.reached)
            if rclpy.ok():
                self.goal_reached_pub.publish(msg)

            if self.reached:
                self.get_logger().info("Spotted goal. Shutting down ...")
                time.sleep(5)
                rclpy.shutdown()

            rate.sleep()

def main(args=None):

    parser = argparse.ArgumentParser(description="ROS 2 Navigation Node with Optional Rotation Mode")
    parser.add_argument('--rotate', action='store_true', help="Enable rotation mode (publishes /vel)")

    known_args, _ = parser.parse_known_args()

    rclpy.init(args=args)
    search_node = Search(rotate=known_args.rotate)

    executor = MultiThreadedExecutor()
    executor.add_node(search_node)

    search_loop = Thread(target=search_node.search_loop)
    search_loop.start()

    try:
        rclpy.spin(search_node, executor=executor)
    except KeyboardInterrupt:
        print("Killing nomad search ...")
        search_loop.join()
        search_node.destroy_node()


if __name__ == "__main__":
    main()
