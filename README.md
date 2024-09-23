## Overview

This readme explains the scripts to manage the setup, execution, tuning, training, and cleanup processes for the project. Below, you will find detailed instructions on how to deploy the model.

- This is a ROS2 adaptation (NoMaD only and focused on Navigation) of the original work by [robodhruv](https://github.com/robodhruv/visualnav-transformer).
- ROS humble and miniconda3 must be installed in `/opt/`.
- `gdown` should be installed to download model waights from google drive.
- You must have a camera node that publish the image topic.
- Your robot must intrepret `/cmd_vel` topic for its motion.
- The files in this repo should be located inside the `src` folder, which should be inside your workspace directory.

---

## `exe.sh`

### Purpose

The `exe.sh` script serves as the main entry point to run other scripts. It provides options to run the setup, execution, tuning the parameters, training the model, and cleaning up repo of this project. All related scripts are located in the `src/nomad/exe` folder.

### Usage

1. **Navigate to the workspace directory:**
   ```bash
   cd /path/to/workspace
   ```

2. **Run the exe script:**
   ```bash
   ./src/exe.sh
   ```

### Script Breakdown

- The `exe.sh` script provides a menu to choose:
  - Setup
  - Run
  - Tune
  - Train
  - Cleanup

---

## `Setup`

### Purpose

The `setup.sh` script sets up the environment and necessary configurations for the project.

### Script Breakdown

- **Setup Paths:**
  Determines the directory of the script and sets the current working path.

- **Create Configuration Files:**
  Generates `path.yaml` and `navigate.yaml` files with project-specific configurations.

- **Run Colcon Build:**
  Compiles the nomad package.

- **Download Model Weights:**
  Downloads the `nomad.pth` from [Google drive](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg).

- **Update Submodule:**
  Updates the `diffusion_policy` [submodule](https://github.com/real-stanford/diffusion_policy).

- **Create Conda Environment:**
  Creates Conda environments (`deploy_nomad` and `train_nomad`) based on YAML configuration files.

- **Install Packages in Environments:**
  Installs required packages in the created Conda environments.

---

## `Tune`

### Purpose

The `tune.sh` script allows users to adjust various parameters of the project by modifying the values in the `navigate.yaml`, `model.yaml`, and `controller.yaml` configuration files. This script provides an interactive menu to choose and update specific parameters.

### Parameters

The script allows tuning of the following parameters:

- **Maximum linear velocity of the model (`v_max`):**
  Controls the maximum linear speed of the model in the navigation process.

- **Maximum angular velocity of the model (`w_max`):**
  Controls the maximum angular speed of the model in the navigation process.

- **Model loop frequency (`hz`):**
  Sets the frequency at which the model's control loop runs.

- **Graph update frequency (`graph_hz`):**
  Determines how often the graph visualization updates.

- **Number of samples (`num_samples`):**
  Specifies the number of samples used in the model's prediction.

- **Radius (`radius`):**
  Defines the radius within which topomap images are compared.

- **Goal node (`goal_node`):**
  Indicates the specific goal node in the topomap.

- **Number of diffusion iterations (`num_diffusion_iters`):**
  Sets the number of iterations for the diffusion process in the model.

- **Length of future predictions (`len_traj_pred`):**
  Defines the length of the trajectory predictions made by the model.

- **Context size (`context_size`):**
  Determines the number of images from the topic that are compared.

- **Maximum linear velocity of the robot (`v_max` in `controller.yaml`):**
  Sets the maximum linear speed of the robot controlled by the PD controller.

- **Maximum angular velocity of the robot (`w_max` in `controller.yaml`):**
  Sets the maximum angular speed of the robot controlled by the PD controller.

- **Controller loop frequency (`frame_rate`):**
  Defines the frequency at which the PD controller's control loop runs.

- **Controller waypoint timeout (`waypoint_timeout`):**
  Specifies the timeout period for the PD controller's waypoints.

- **Waypoint (`waypoint`):**
  Sets the specific waypoint for navigation.

- **Close threshold (`close_threshold`):**
  Defines the threshold distance for considering the goal as reached in the navigation process.

### Script Breakdown

Upon `tune.sh` selection, the current value of the parameter is displayed, and users are prompted to enter a new value. The script then updates the respective YAML configuration file with the new value.

---

## `Run`

### Purpose

The `run.sh` script is designed to collect new trajectories, create data from recorded bags, generate topomaps, and navigate with the trajectory.

### Script Breakdown

- **Bag Name Input:**
  Prompts the user to enter a name for the bag file.

- **Collect Trajectory:**
  Prompts the user to start recording a new bag file and handle data collection.

- **Create Training Data:**
  Uses the recorded bag to create training data and displays it.

- **Create Topomap:**
  Uses the recorded bag to generate topomap.

- **Navigate:**
  Initiates the navigation process using the collected trajectory and created topomap.

### Key Commands and Parameters

- **Collect Trajectory:**
  Records image and odometry data into a ROS bag file.
  ```bash
  ros2 bag record /cam0/image_raw /odom/local -o src/nomad/preprocessing/rosbags/$bag_name
  ```
  - `-o`: Output path for the bag file.
  - `/cam0/image_raw`: Topic for image data.
  - `/odom/local`: Topic for odometry data.
  - **Explanation:** This command records the data from the specified topics (`/cam0/image_raw` and `/odom/local`) and saves it in a ROS bag file named `$bag_name` in the specified output directory.

- **Create Training Data:**
  Processes the recorded bag to generate training data.
  ```bash
  python3 src/nomad/preprocessing/process_bag_diff.py -i src/nomad/preprocessing/rosbags/$bag_name -o src/nomad/preprocessing/training_data -n -1 -s 4.0 -c /cam0/image_raw -d /odom/local
  ```
  - `-i`: Input path to the recorded bag file.
  - `-o`: Output path for the training data.
  - `-n`: Number of data points to generate.
  - `-s`: Step size for sampling data.
  - `-c`: Image topic.
  - `-d`: Odom topic.
  - **Explanation:** This command processes the recorded bag file to generate training data, using the specified parameters such as the number of data points, step size, image topic, and odom topic.

  ```bash
  python3 src/nomad/preprocessing/pickle_data.py -f src/nomad/preprocessing/training_data/${bag_name}_0/traj_data.pkl -g
  ```
  - `-f`: Path to the trajectory data pickle file.
  - `-g`: Flag to plot the graph of positions (x, y) and yaw.
  - **Explanation:** This command plots the graph of positions and yaw of the processed data, ensuring it is collected without any error.

- **Create Topomap:**
  Generates a topological map of the environment from the images recorded in the bag file.
  ```bash
  ros2 run nomad create_topomap.py -b src/nomad/preprocessing/rosbags/$bag_name -T src/nomad/preprocessing/topomap -d $bag_name -i /cam0/image_raw -t 1.0 -w 1
  ```
  - `-b`: Path to the bag file.
  - `-T`: Path to the topomap directory.
  - `-d`: Directory name for the topomap images.
  - `-i`: Image topic.
  - `-t`: Time interval between images.
  - `-w`: Number of worker threads.
  - **Explanation:** This command creates a topological map from the images recorded in the specified bag file. The images are saved in the specified topomap directory, with a specified time interval between them. The number of worker threads used for processing is also specified.

- **Navigate:**
  Runs the PD controller and the navigation script to start navigating based on the created topomap.
  ```bash
  ros2 run nomad pd_controller.py
  ```
  - **Explanation:** This command starts the Proportional-Derivative (PD) controller which is responsible for driving the robot based on the waypoints generated.

  ```bash
  ros2 run nomad navigate.py --ros-args --params-file src/nomad/deploy/config/navigate.yaml --remap /img:=/cam0/image_raw
  ```
  - `--ros-args`: Passes ROS arguments.
  - `--params-file`: Path to the parameter file.
  - `--remap /img:=/cam0/image_raw`: Remaps the image topic.
  - **Explanation:** This command runs the navigation script, which uses the PD controller and the generated topomap to navigate through the environment. The image topic is remapped to ensure the correct image data is used during navigation.

---

## `Train`

### Purpose

The `train.sh` script is used to train the Nomad model. It allows for training from scratch or continuing training from a previous checkpoint.

### Script Breakdown

- **Training Mode Prompt:**
  Prompts the user to choose whether to train from scratch or continue training from a previous checkpoint.

- **Configuration Adjustments:**
  Comments or uncomments the `load_run` line in the `model.yaml` configuration file based on the user's choice.

- **Activate Conda Environment:**
  Activates the `train_nomad` Conda environment.

- **Run Data Split and Training:**
  Executes the `data_split.py` and `train.py` scripts to split the data and train the model.

### Commands

- **Data Split:**
  ```bash
  python3 data_split.py
  ```

- **Training:**
  ```bash
  python3 train.py
  ```

### Parameters

- **train_from_scratch:** Determines whether to train from scratch or continue from a previous checkpoint.
- **model_weights_path:** Path to the model weights file.
- **model_config_path:** Path to the model configuration file.

---

## `Cleanup`

### Purpose

The `cleanup.sh` script is used to clean up the environment by removing collected data, training logs, and uninstallation of environments and dependencies. This will reset the repository to its initial state.

### Options

1. **Delete Collected Data:**
   - Removes all collected data including rosbags, topomaps, and training data.

2. **Clear Training Logs:**
   - Clears all training logs, including wandb cache and data splits.

3. **Select What to Remove:**
   - Allows the user to select specific items to remove, such as pycache, deploy config files, diffusion policy, etc.

4. **Cleanup:**
   - Removes everything related to the project from the workspace, ensuring the workspace is in a state similar to a freshly cloned repository.

5. **Uninstall:**
   - Removes everything related to the project from the system, including Conda environments, ensuring the system is in a state similar to a freshly cloned repository.
