import numpy as np
import io
import os
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import torchvision.transforms.functional as TF
from sensor_msgs.msg import Image as RosImage, CompressedImage
from nav_msgs.msg import Odometry
import transforms3d.euler
import numpy as np
from scipy.interpolate import interp1d
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3

def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in im_list:
        img = img_process_func(img_msg)
        images.append(img)
    print(f"Processed {len(images)} images")
    return images

def process_tartan_img(msg: RosImage) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the tartan_drive dataset
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    img = np.moveaxis(img, 0, -1)  # reverse the axis order to get the image in the right orientation
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # convert rgb to bgr
    img = Image.fromarray(img)
    return img

def process_locobot_img(msg: RosImage) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image

def process_scand_img(msg: CompressedImage) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    img = Image.open(io.BytesIO(msg.data))  # convert sensor_msgs/CompressedImage to PIL image
    w, h = img.size
    img = TF.center_crop(img, (h, int(h * IMAGE_ASPECT_RATIO)))  # center crop image to 4:3 aspect ratio
    img = img.resize(IMAGE_SIZE)  # resize image to IMAGE_SIZE
    return img

def process_sacson_img(msg: CompressedImage) -> Image:
    np_arr = np.frombuffer(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image

def process_odom(odom_list: List, odom_process_func: Any, ang_offset: float = 0.0) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in odom_list:
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    print(f"Processed {len(xys)} odom messages")
    return {"position": np.array(xys), "yaw": np.array(yaws)}

def nav_to_xy_yaw(odom_msg: Odometry, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """
    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w) + ang_offset
    return [position.x, position.y], yaw

# def get_images_and_odom(reader, imtopics, odomtopics, rate=4.0):
#     """
#     Get image and odom data from a ROS2 bag file using rosbag2_py

#     Args:
#         reader (rosbag2_py.SequentialReader): rosbag2 reader
#         imtopics (list[str]): topic name(s) for image data
#         odomtopics (list[str]): topic name(s) for odom data
#         img_process_func (Any): function to process image data
#         odom_process_func (Any): function to process odom data
#         rate (float, optional): rate to sample data. Defaults to 4.0.
#     Returns:
#         img_data (dict): dictionary of lists of PIL images
#         traj_data (dict): dictionary of lists of odom data
#     """
#     from rclpy.serialization import deserialize_message
#     from rosidl_runtime_py.utilities import get_message

#     topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}

#     def create_msg_type(topic):
#         return get_message(topic_types[topic])

#     im_msgs = {topic: [] for topic in imtopics}
#     odom_msgs = {topic: [] for topic in odomtopics}
#     currtime = None

#     while reader.has_next():
#         topic, data, t = reader.read_next()
#         msg = deserialize_message(data, create_msg_type(topic))

#         if topic in imtopics:
#             im_msgs[topic].append((msg, t))
#         elif topic in odomtopics:
#             odom_msgs[topic].append((msg, t))

#         if currtime is None:
#             currtime = t / 1e9  # Convert nanoseconds to seconds

#         current_t = t / 1e9  # Convert nanoseconds to seconds
#         if (current_t - currtime) >= 1.0 / rate:
#             currtime = current_t

#     if not im_msgs or not odom_msgs:
#         return None, None

#     bag_img_data = {topic: process_images([msg for msg, _ in im_msgs[topic]], process_tartan_img) for topic in imtopics}
#     bag_traj_data = {topic: process_odom([msg for msg, _ in odom_msgs[topic]], nav_to_xy_yaw) for topic in odomtopics}

#     return bag_img_data, bag_traj_data

def get_images_and_odom(reader, imtopics, odomtopics, rate=4.0):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}

    def create_msg_type(topic):
        return get_message(topic_types[topic])

    im_msgs = {topic: [] for topic in imtopics}
    odom_msgs = {topic: [] for topic in odomtopics}
    currtime = None

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg = deserialize_message(data, create_msg_type(topic))

        if topic in imtopics:
            im_msgs[topic].append((msg, t))
        elif topic in odomtopics:
            odom_msgs[topic].append((msg, t))

        if currtime is None:
            currtime = t / 1e9

        current_t = t / 1e9
        if (current_t - currtime) >= 1.0 / rate:
            currtime = current_t

    if not im_msgs or not odom_msgs:
        return None, None

    print(f"Number of raw image messages: {sum(len(msgs) for msgs in im_msgs.values())}")
    print(f"Number of raw odom messages: {sum(len(msgs) for msgs in odom_msgs.values())}")

    bag_img_data = {topic: process_images([msg for msg, _ in im_msgs[topic]], process_tartan_img) for topic in imtopics}

    def scale_down_odom(odom_list, num_images, trim=False):
        if trim:
            num_to_remove = len(odom_list) // 50
            trimmed_odom_list = odom_list[num_to_remove:-num_to_remove]
            print(f"Trimmed odom messages from {len(odom_list)} to {len(trimmed_odom_list)}")
        else:
            trimmed_odom_list = odom_list

        indices = np.round(np.linspace(0, len(trimmed_odom_list) - 1, num_images)).astype(int)
        scaled_odom = [trimmed_odom_list[i] for i in indices]
        print(f"Scaled odom messages from {len(trimmed_odom_list)} to {len(scaled_odom)}")
        return scaled_odom


    scaled_odom_msgs = {topic: scale_down_odom([msg for msg, _ in odom_msgs[topic]], len(bag_img_data[imtopics[0]])) for topic in odomtopics}

    bag_traj_data = {topic: process_odom(scaled_odom_msgs[topic], nav_to_xy_yaw) for topic in odomtopics}

    return bag_img_data, bag_traj_data

def reverse_rgb(image):
    """Reverse the RGB channels of an image."""
    r, g, b = image.split()
    return Image.merge("RGB", (b, g, r))

# def get_images_and_odom(reader, imtopics, odomtopics, rate=4.0):
#     from rclpy.serialization import deserialize_message
#     from rosidl_runtime_py.utilities import get_message

#     topic_types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}

#     def create_msg_type(topic):
#         return get_message(topic_types[topic])

#     im_msgs = {topic: [] for topic in imtopics}
#     odom_msgs = {topic: [] for topic in odomtopics}
#     currtime = None

#     while reader.has_next():
#         topic, data, t = reader.read_next()
#         msg = deserialize_message(data, create_msg_type(topic))

#         if topic in imtopics:
#             im_msgs[topic].append((msg, t))
#         elif topic in odomtopics:
#             odom_msgs[topic].append((msg, t))

#         if currtime is None:
#             currtime = t / 1e9

#         current_t = t / 1e9
#         if (current_t - currtime) >= 1.0 / rate:
#             currtime = current_t

#     if not im_msgs or not odom_msgs:
#         return None, None

#     print(f"Number of raw image messages: {sum(len(msgs) for msgs in im_msgs.values())}")
#     print(f"Number of raw odom messages: {sum(len(msgs) for msgs in odom_msgs.values())}")

#     bag_img_data = {topic: process_images([msg for msg, _ in im_msgs[topic]], process_tartan_img) for topic in imtopics}

#     def scale_down_odom(odom_list, num_images):
#         indices = np.round(np.linspace(0, len(odom_list) - 1, num_images)).astype(int)
#         scaled_odom = [odom_list[i] for i in indices]
#         print(f"Scaled odom messages from {len(odom_list)} to {len(scaled_odom)}")
#         return scaled_odom

#     scaled_odom_msgs = {topic: scale_down_odom([msg for msg, _ in odom_msgs[topic]], len(bag_img_data[imtopics[0]])) for topic in odomtopics}

#     bag_traj_data = {topic: process_odom(scaled_odom_msgs[topic], nav_to_xy_yaw) for topic in odomtopics}

#     return bag_img_data, bag_traj_data


def is_backwards(pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point
    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps

def filter_backwards(img_list: List[Image.Image], traj_data: Dict[str, np.ndarray], start_slack: int = 0, end_slack: int = 0) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cut out non-positive velocity segments of the trajectory
    Args:
        traj_type: type of trajectory to cut
        img_list: list of images
        traj_data: dictionary of position and yaw data
        start_slack: number of points to ignore at the start of the trajectory
        end_slack: number of points to ignore at the end of the trajectory
    Returns:
        cut_trajs: list of cut trajectories
        start_times: list of start times of the cut trajectories
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair: list) -> Tuple[List, Dict]:
        new_img_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [(img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append((img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]]))
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
    return cut_trajs

def quat_to_yaw(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw

def ros_to_numpy(msg: RosImage, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"):
    """
    Convert a ROS image message to a numpy array
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256 ** i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256 ** i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (255.0 if aggregate == "none" else 255.0 ** nchannels)

    return data

