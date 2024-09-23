#!/usr/bin/env bash

# Ask for bag name
echo "Enter bag name: "
read bag_name

# Define variables for directories and topic names
navigate_config="src/nomad/deploy/config/navigate.yaml"
controller_config="src/nomad/deploy/config/controller.yaml"
rosbag_dir="src/nomad/preprocessing/rosbags/$bag_name"
training_data_dir="src/nomad/preprocessing/training_data"
topomap_dir="src/nomad/preprocessing/topomap"
cam_topic="/cam0/image_raw"
odom_topic="/odom/local"

# Function to setup session
setup() {
    local env_name=$1
    local session_name=$2
    local sleep_time=$3
    echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate $env_name && source /opt/ros/humble/setup.bash && source install/setup.bash && echo 'starting $session_name...' && sleep $sleep_time"
}

# Function to cleanup session
cleanup() {
    local session_name=$1
    echo "echo '...$session_name stopping' && conda deactivate && sleep 3 && tmux kill-session -t $session_name"
}

# Function to create a tmux session
create_tmux_session() {
    local session_name=$1
    local window_name=$2
    local commands=$3

    tmux new-session -d -s "$session_name" -n "$window_name" bash -c "$commands"
    tmux attach -t "$session_name:$window_name"
}

collect_trajectory() {
    local commands="
        $(setup deploy_nomad record_bag 5)
        ros2 bag record $cam_topic $odom_topic -o $rosbag_dir
        $(cleanup record_bag)
    "
    create_tmux_session "record_bag" "record" "$commands"
}

create_training_data() {
    local commands="
        $(setup deploy_nomad data_collection 0)
        python3 src/nomad/preprocessing/process_bag_diff.py -i $rosbag_dir -o $training_data_dir -n -1 -s 4.0 -c $cam_topic -d $odom_topic
        python3 src/nomad/preprocessing/pickle_data.py -f $training_data_dir/${bag_name}_0/traj_data.pkl -g
        $(cleanup data_collection)
    "
    create_tmux_session "data_collection" "collection" "$commands"
}

create_topomap() {
    local commands="
        $(setup deploy_nomad topomap_creation 0)
        ros2 run nomad create_topomap.py -b $rosbag_dir -T $topomap_dir -d $bag_name -i $cam_topic -t 1.0 -w 1
        $(cleanup topomap_creation)
    "
    create_tmux_session "topomap_creation" "topomap" "$commands"
}

navigate() {
    tmux new-session -d -s navigation -n navigator bash -c "
        $(setup deploy_nomad controller 0)
        ros2 run nomad pd_controller.py --ros-args --params-file $controller_config
    "
    tmux split-window -v -t navigation:navigator bash -c "
        $(setup deploy_nomad navigation 5)
        sed -i 's|topomap/[^\"]*|topomap/$bag_name|' $navigate_config
        ros2 run nomad navigate.py --ros-args --params-file $navigate_config --remap /img:=$cam_topic
        $(cleanup navigation)
    "
    tmux attach -t navigation
}

# Main menu function
main_menu() {
    while true; do
    	clear
        echo "Choose an option:"
        echo "1. Collect trajectory"
        echo "2. Create training data"
        echo "3. Create topomap"
        echo "4. Navigate"
        echo "9. Back"
        echo "0. Exit"

        read -p "Enter your choice: " choice
        echo    # move to a new line

        case $choice in
            1)
                collect_trajectory
                ;;
            2)
                create_training_data
                ;;
            3)
                create_topomap
                ;;
            4)
                navigate
                ;;
            9)
                ./src/exe.sh
                break
                ;;
            0)
                echo "Exiting."
                sleep 1
                clear
                exit 0
                ;;
            *)
                echo "Invalid choice. Please select a valid option."
                sleep 2
                ;;
        esac
    done
}

# Start the script
main_menu
