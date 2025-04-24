#!/usr/bin/env bash

# Ask for bag name
echo "Enter bag name: "
read bag_name

# Define variables for directories and topic names
model_config="src/nomad/deploy/config/nomad.yaml"
opt_config="src/nomad/deploy/config/model.yaml"
controller_config="src/nomad/deploy/config/controller.yaml"
weight="src/nomad/deploy/model_weights/nomad.pth"
rosbag_dir="src/nomad/preprocessing/rosbags/$bag_name"
training_data_dir="src/nomad/preprocessing/training_data"
back_target_dir="src/nomad/preprocessing/map/backward/$bag_name"
for_target_dir="src/nomad/preprocessing/map/forward/$bag_name"
target_dir="src/nomad/preprocessing/target"
backward_dir="src/nomad/preprocessing/map/backward"
forward_dir="src/nomad/preprocessing/map/forward"
topomap="src/nomad/preprocessing/topomap/$bag_name"
forward_cam_topic="forward/image_raw"
backward_cam_topic="backward/image_raw"
odom_topic="/odom_topic"
vel_topic="/cmd_vel"

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
    echo "echo '...$session_name stopping' && conda deactivate && tmux kill-session -t $session_name"
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
        rm -rf $rosbag_dir
        ros2 bag record $forward_cam_topic $backward_cam_topic $odom_topic -o $rosbag_dir
        $(cleanup record_bag)
    "
    create_tmux_session "record_bag" "record" "$commands"
}

create_training_data() {
    local commands="
        $(setup deploy_nomad data_collection 0)
        python3 src/nomad/preprocessing/process_bag_diff.py -i $rosbag_dir -o $training_data_dir -n -1 -s 4.0 -c $forward_cam_topic -d $odom_topic
        python3 src/nomad/preprocessing/pickle_data.py -f $training_data_dir/${bag_name}_0/traj_data.pkl -g
        $(cleanup data_collection)
    "
    create_tmux_session "data_collection" "collection" "$commands"
}

create_topomap() {
    local mode=${1:-forward}

    if [[ "$mode" == "backward" ]]; then
        local commands="
            $(setup deploy_nomad topomap_creation 0)
            ros2 run nomad create_topomap.py -b $rosbag_dir -T $backward_dir -d $bag_name -i $backward_cam_topic -t 1.0 -w 1 --reverse
            python3 src/nomad/deploy/code/deployment/optimize.py -d $back_target_dir -n $weight -o $topomap -y $opt_config
            mkdir -p $target_dir && cp $back_target_dir/0.png $target_dir
            $(cleanup topomap_creation)
        "
    else
        local commands="
            $(setup deploy_nomad topomap_creation 0)
            ros2 run nomad create_topomap.py -b $rosbag_dir -T $forward_dir -d $bag_name -i $forward_cam_topic -t 1.0 -w 1
            python3 src/nomad/deploy/code/deployment/optimize.py -d $for_target_dir -n $weight -o $topomap -y $opt_config
            mkdir -p $target_dir && cp $for_target_dir/0.png $target_dir
            $(cleanup topomap_creation)
        "
    fi

    create_tmux_session "topomap_creation" "topomap" "$commands"
}

run_controller() {
    echo "ros2 run nomad controller.py --ros-args --params-file $controller_config --remap /vel:=$vel_topic"
}

navigate() {
    tmux new-session -d -s navigation -n navigator bash -c "
        $(setup deploy_nomad controller 0)
        $(run_controller)
    "
    tmux split-window -v -t navigation:navigator bash -c "
        $(setup deploy_nomad navigation 5)
        sed -i 's|topomap/[^\"]*|topomap/$bag_name|' $model_config
        ros2 run nomad navigate.py --ros-args --params-file $model_config --remap /img:=$forward_cam_topic
        $(cleanup navigation)
    "
    tmux attach -t navigation
}

explore() {
    tmux new-session -d -s exploration -n explorer bash -c "
        $(setup deploy_nomad record_bag 5)
        rm -rf $rosbag_dir
        ros2 bag record $backward_cam_topic $forward_cam_topic $odom_topic -o $rosbag_dir
        $(cleanup exploration)
    "
    tmux split-window -v -t exploration:explorer -p 80 bash -c "
        $(setup deploy_nomad controller 0)
        $(run_controller)
    "
    tmux split-window -v -t exploration:explorer bash -c "
        $(setup deploy_nomad exploration 5)
        ros2 run nomad explore.py --ros-args --params-file $model_config --remap /img:=$forward_cam_topic
    "
    tmux attach -t exploration
}

search() {
    tmux new-session -d -s search -n searcher bash -c "
        $(setup deploy_nomad controller 0)
        $(run_controller)
    "
    tmux split-window -v -t search:searcher bash -c "
        $(setup deploy_nomad search 5)
        ros2 run nomad search.py --ros-args --params-file $model_config --remap /img:=$forward_cam_topic
        exec bash
        $(cleanup search)
    "
    tmux attach -t search
}
boomerang() {
    explore
    tmux wait-for exploration_done

    create_topomap backward
    tmux wait-for topomap_done

    tmux new-session -d -s search -n searcher bash -c "
        $(setup deploy_nomad rotation 0)
        ros2 topic echo $vel_topic geometry_msgs/msg/Twist
        tmux wait-for -S search_done
    "
    tmux split-window -v -t search:searcher bash -c "
        $(setup deploy_nomad search 0)
        ros2 run nomad search.py --rotate --ros-args --params-file $model_config --remap /img:=$forward_cam_topic --remap /vel:=$vel_topic
        $(cleanup search)
    "
    tmux attach -t search
    tmux wait-for search_done

    navigate
    tmux wait-for navigation_done
}

# Main menu function
main_menu() {
    while true; do
    	clear
    	echo "Bag Name: $bag_name"
        echo "Choose an option:"
        echo "1. Collect trajectory"
        echo "2. Create training data"
        echo "3. Create topomap"
        echo "4. Navigate"
        echo "5. Explore"
        echo "6. Search"
        echo "7. Boomerang"
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
            5)
                explore
                ;;
            6)
                search
                ;;
            7)
                boomerang
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
