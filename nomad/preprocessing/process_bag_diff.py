import os
import pickle
from PIL import Image
import argparse
import tqdm
import yaml
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# utils
from process_data_utils import get_images_and_odom, reverse_rgb

def main(args: argparse.Namespace):
    # Load the config file if needed
    # with open("/home/flo/flo_nav_ws/my_bag/process_bag/process_bags.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Path to metadata.yaml
    metadata_path = os.path.join(args.input_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        print(f"Metadata file not found: {metadata_path}")
        return

    # Load metadata.yaml
    with open(metadata_path, 'r') as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)

    # Access relative file paths correctly
    try:
        db3_path = os.path.join(args.input_dir, metadata['rosbag2_bagfile_information']['relative_file_paths'][0])
    except KeyError as e:
        print(f"Key {e} not found in metadata.yaml")
        return

    if not os.path.exists(db3_path):
        print(f"Bag file not found: {db3_path}")
        return

    bag_files = [db3_path]

    # Processing loop
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
            converter_options = rosbag2_py.ConverterOptions('', '')
            reader = rosbag2_py.SequentialReader()
            reader.open(storage_options, converter_options)
        except Exception as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # Create a name for the trajectory using the filename without the extension
        traj_name = os.path.splitext(os.path.basename(bag_path))[0]

        # Load the bag file
        bag_img_data, bag_traj_data = get_images_and_odom(
            reader,
            [args.camera_topic],  # Updated topic name for image
            [args.odom_topic],  # Topic name for odometry
            rate=args.sample_rate,
        )

        if bag_img_data is None:
            print(f"{bag_path} did not have the topics we were looking for. Skipping...")
            continue

        traj_folder = os.path.join(args.output_dir, traj_name)
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)

        obs_images = bag_img_data[args.camera_topic]
        for i, obs_image in enumerate(obs_images):
            obs_image = reverse_rgb(obs_image)
            obs_image.save(os.path.join(traj_folder, f"{i}.jpg"))

        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(bag_traj_data[args.odom_topic], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="Path of the datasets with rosbags",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./topomap/",
        type=str,
        help="Path for processed dataset (default: ./topomap/)",
    )
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="Number of bags to process (default: -1, all)",
    )
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="Sampling rate (default: 4.0 hz)",
    )
    parser.add_argument(
        "--camera-topic",
        "-c",
        type=str,
        help="Camera topic to process",
        required=True,
    )
    parser.add_argument(
        "--odom-topic",
        "-d",
        type=str,
        help="Odometry topic to process",
        required=True,
    )

    args = parser.parse_args()
    print("STARTING PROCESSING DIFF DATASET")
    main(args)
    print("FINISHED PROCESSING DIFF DATASET")
