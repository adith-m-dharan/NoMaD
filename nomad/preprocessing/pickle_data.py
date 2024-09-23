import argparse
import pickle
import os
import matplotlib.pyplot as plt

def print_data(start, end, file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        positions = data["position"]
        yaws = data["yaw"]

        if start < 0 or end >= len(positions):
            print("Error: The specified range is out of bounds.")
            return

        for i in range(start, end + 1):
            print(f"Robot position at image {i + 1}: {positions[i]}")
            print(f"Robot yaw at image {i + 1}: {yaws[i]} (degrees)")

    except FileNotFoundError:
        print(f"Error: Could not find {file_path} file!")

def plot_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        positions = data["position"]
        yaws = data["yaw"]

        x_vals = [pos[0] for pos in positions]
        y_vals = [pos[1] for pos in positions]
        yaw_vals = list(yaws)
        img_nums = list(range(1, len(positions) + 1))

        plt.figure(figsize=(15, 5))
        plt.suptitle(f"Data from {file_path}", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.plot(img_nums, x_vals, label='X Position')
        plt.xlabel('Image Number')
        plt.ylabel('X Position')
        plt.title('X Position vs Image Number')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(img_nums, y_vals, label='Y Position')
        plt.xlabel('Image Number')
        plt.ylabel('Y Position')
        plt.title('Y Position vs Image Number')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(img_nums, yaw_vals, label='Yaw')
        plt.xlabel('Image Number')
        plt.ylabel('Yaw (degrees)')
        plt.title('Yaw vs Image Number')
        plt.legend()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find {file_path} file!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print robot position and yaw for a range of images or plot graphs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--print', nargs=2, type=int, metavar=('START', 'END'), help="Print position and yaw from START to END index (inclusive).")
    group.add_argument('-g', '--graph', action='store_true', help="Plot graphs for the entire range of data.")
    parser.add_argument('-f', '--file', type=str, default='traj_data.pkl', help="Path to the .pkl file (default: 'traj_data.pkl' in current directory).")

    args = parser.parse_args()
    file_path = args.file

    if args.print:
        start, end = args.print
        print_data(start, end, file_path)
    elif args.graph:
        plot_data(file_path)
