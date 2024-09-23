#!/usr/bin/env python3

import os
import argparse
import threading
from threading import Thread
import multiprocessing
import shutil

import rclpy
from rclpy.node import Node

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

from tqdm import tqdm
from utils import msg_to_pil, Rate, is_valid_ros2_bag, split_list


class CreateTopomap(Node):
    def __init__(self, args):
        super().__init__('create_topomap')
        self.path_to_ros2_bag = args.path_to_ros2_bag
        self.dt = args.dt
        self.dir = args.dir
        self.image_topic = args.image_topic
        self.topomap_images_dir = args.topomap_images_dir
        self.workers = args.workers

        max_workers = multiprocessing.cpu_count()

        if self.workers > max_workers:
            self.get_logger().warn(f"Inputted worker count exceeds cpu count. \
                                   Defaulting to {max_workers}")
            self.workers = max_workers

        assert is_valid_ros2_bag(
            self.path_to_ros2_bag), "path to ros2 bag must be a valid bag"
        assert self.topomap_images_dir != "", "path to topomap dir must not be empty"

        self.obs_img = None

        self.typestore = get_typestore(Stores.ROS2_HUMBLE)

        self.reader = Reader(self.path_to_ros2_bag)
        self.reader.open()

        self.topomap_name_dir = os.path.join(self.topomap_images_dir, self.dir)
        if not os.path.isdir(self.topomap_name_dir):
            os.makedirs(self.topomap_name_dir)
        else:
            self.get_logger().info(
                f"{self.topomap_name_dir} already exists. Removing previous images...")
            self.remove_files_in_dir(self.topomap_name_dir)

        self.rate = Rate(1 / self.dt)

    def remove_files_in_dir(self, dir_path: str):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                self.get_logger().error(
                    f"Failed to delete {file_path}. Reason: {e}")

    def create_img_msgs_list(self):
        self.img_msgs = []
        start = self.get_clock().now()
        filtered_connections = [conn for conn in self.reader.connections \
                                if conn.topic == self.image_topic]
        idx = 0
        for conn, _, rawdata in self.reader.messages(connections=filtered_connections):
            msg = self.typestore.deserialize_cdr(rawdata, conn.msgtype)
            self.img_msgs.append((idx, msg))
            idx += 1

        elapsed = self.get_clock().now() - start
        elapsed = elapsed.nanoseconds/1e9
        print(f"Elapsed time for {len(self.img_msgs)} : {elapsed} secs")

    def worker_job(self, img_list):
        for idx, img_msg in tqdm(img_list, leave=True):
            if not rclpy.ok():
                break
            img = msg_to_pil(img_msg)
            img.save(os.path.join(
                self.topomap_name_dir, f"{idx}.png"))
            
        self.worker_threads[threading.current_thread().name]['done'] = True
            
    def spawn_workers(self):
        self.get_logger().info(f"Spawning {self.workers} worker threads")
        self.worker_threads = {}
        msg_lists = split_list(self.img_msgs, self.workers)
        for i in range(self.workers):
            thread = Thread(target=self.worker_job, args=(msg_lists[i],), name=f"worker-{i}")
            thread.start()
            worker = {
                'thread': thread,
                'done': False
            }
            self.worker_threads[thread.name] = worker

    def wait_for_workers(self):
        rate = Rate(hz=1)
        while rclpy.ok():
            done = True
            for _, worker in self.worker_threads.items():
                thread = worker["thread"]
                done = done and worker["done"]
                thread.join(timeout=1)

            if done:
                rclpy.shutdown()

            rate.sleep()

    def teardown(self):
        self.reader.close()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(
        description="Code to generate topomaps from the image topic"
    )
    parser.add_argument(
        "--path-to-ros2-bag",
        "-b",
        required=True,
        type=str,
        help="path to input ros2 bag",
    )
    parser.add_argument(
        "--topomap_images_dir",
        "-T",
        required=True,
        type=str,
        help="path to topomap collections",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topological map images in directory (default: topomap)",
    )
    parser.add_argument(
        "--dt",
        "-t",
        default=1.0,
        type=float,
        help="time between images sampled from the image topic (default: 1.0)",
    )
    parser.add_argument(
        "--image_topic",
        "-i",
        default="/image_raw",
        type=str,
        help="image topic to subscribe to",
    )
    parser.add_argument(
        "--workers",
        "-w",
        default=1,
        type=int,
        help="parallel worker threads. max is cpu count"
    )
    args = parser.parse_args()

    create_topomap = CreateTopomap(args)
    create_topomap.create_img_msgs_list()
    create_topomap.spawn_workers()

    try:
        create_topomap.wait_for_workers()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        create_topomap.teardown()
        create_topomap.destroy_node()


if __name__ == "__main__":
    main()
