#!/usr/bin/env bash

# Define the tasks array
tasks=(
    "run_colcon_build:Do you want to build ROS package?:ROS package build"
    "download_nomad_pth:Do you want to download model weight?:Weight download"
    "git submodule update --init --recursive:Do you want to update submodule?:Submodule update"
    "create_conda_environment deploy_nomad nomad/deploy/deploy_nomad.yaml:Do you want to create deployment environment?:Create deploy env"
    "create_conda_environment train_nomad nomad/train/train_nomad.yaml:Do you want to create training environment?:Create train env"
    "install_packages_in_envs:Do you want to install packages in environments?:Package install"
)

# Function to confirm and execute a command
confirm_and_execute() {
    local prompt_message=$1
    local command=$2
    local task_name=$3
    read -p "$prompt_message (y/n): " -n 1 -r
    echo    # move to a new line
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        eval $command
        echo "$task_name completed."
    else
        echo "Skipped $task_name."
    fi
}

# Function to execute tasks based on provided indices and confirmation flag
execute_tasks() {
    local confirm_tasks=${!#}
    local indices=("${@:1:$(($#-1))}")

    for index in "${indices[@]}"; do
        IFS=":" read -r command prompt_message task_name <<< "${tasks[$index]}"
        if [ "$confirm_tasks" = true ]; then
            confirm_and_execute "$prompt_message" "$command" "$task_name"
        else
            eval "$command"
            echo "$task_name completed."
        fi
    done
}

# Get the directory of the script
setup_paths() {
    SCRIPT_DIR=$(dirname "$0")
    cd "$SCRIPT_DIR"
    cd ../..
    current_path=$(pwd)
}

# Create configuration files
create_config_files() {
    local output_path="$current_path/nomad/train/config/path.yaml"
    local dataset_content="datasets:
  training_data:
    data_folder: $current_path/nomad/preprocessing/training_data/
    train: $current_path/nomad/preprocessing/data_splits/training_data/train/
    test: $current_path/nomad/preprocessing/data_splits/training_data/test/"
    mkdir -p "$(dirname "$output_path")"
    echo "$dataset_content" > "$output_path"

    local navigate_output_path="$current_path/nomad/deploy/config/navigate.yaml"
    local navigate_content="nomad_navigator:
  ros__parameters:
    model_name: \"nomad\"
    model_weights_path: \"$current_path/nomad/deploy/model_weights/nomad.pth\"
    model_config_path: \"$current_path/nomad/train/config/model.yaml\"
    topomap_images_dir: \"$current_path/nomad/preprocessing/topomap\"
    topomap_dir: \"$current_path/nomad/preprocessing/topomap/bag_name\"
    waypoint: 2
    goal_node: -1
    close_threshold: 3
    radius: 4
    skip: 1
    tolerence: 0
    num_samples: 8
    v_max: 0.2
    w_max: 0.2
    hz: 4.0
    graph_hz: 0.33"
    mkdir -p "$(dirname "$navigate_output_path")"
    echo "$navigate_content" > "$navigate_output_path"
    
    echo "Configs created"
}

# Run colcon build if needed
run_colcon_build() {
    tmux new-session -d -s colcon_build -c "$current_path/.." "
        source /opt/miniconda3/etc/profile.d/conda.sh
        conda deactivate
        source /opt/ros/humble/setup.bash
        colcon build --symlink-install
        echo 'Colcon build completed.'
        sleep 3
        tmux kill-session -t colcon_build
    "
    tmux attach -t colcon_build
}

# Download nomad.pth if needed
download_nomad_pth() {
    FILE_ID="1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb"
    FILE_NAME="nomad.pth"
    DOWNLOAD_DIR="$current_path/nomad/deploy/model_weights"

    echo "Getting model weights.."
    mkdir -p "${DOWNLOAD_DIR}"
    pip install gdown > /dev/null 2>&1
    gdown ${FILE_ID} -O "${DOWNLOAD_DIR}/${FILE_NAME}"
    pip uninstall gdown -y > /dev/null 2>&1
}

# Create Conda environment if needed
create_conda_environment() {
    local env=$1
    local yaml_file=$2

    source /opt/miniconda3/etc/profile.d/conda.sh
    conda clean --all -y
    if conda env list | grep -q "^$env\s"; then
        confirm_and_execute "$env already exists. Do you want to replace it?" REPLACE_ENV
        if [ "$REPLACE_ENV" = true ]; then
            conda remove --name "$env" --all
            conda env create -f "$yaml_file"
            echo "Conda environment $env created."
        else
            echo "Skipping creation of $env environment."
        fi
    else
        conda env create -f "$yaml_file"
        echo "Conda environment $env created."
    fi
    conda clean --all -y
}

# Install packages in specified environments
install_packages_in_envs() {
    SESSION_NAME="install_pkg"
    env_list=("train_nomad" "deploy_nomad")

    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "tmux session $SESSION_NAME already exists"
    else
        tmux new-session -d -s "$SESSION_NAME"

        for env in "${env_list[@]}"; do
            if conda env list | grep -q "^$env\s"; then
                tmux send-keys -t "$SESSION_NAME" "source /opt/miniconda3/etc/profile.d/conda.sh" C-m
                tmux send-keys -t "$SESSION_NAME" "conda activate $env" C-m
                tmux send-keys -t "$SESSION_NAME" "pip install -e $current_path/nomad/diffusion_policy && pip install -e $current_path/nomad/train" C-m
                tmux send-keys -t "$SESSION_NAME" "conda deactivate" C-m
            else
                echo "Conda environment $env does not exist. Skipping package installation for $env."
            fi
        done

        tmux send-keys -t "$SESSION_NAME" "sleep 3; tmux kill-session -t $SESSION_NAME" C-m
        tmux attach -t "$SESSION_NAME"
    fi
}

# Main script execution
installation() {
    setup_paths
    create_config_files
    execute_tasks "$@"
}

# Main menu function
main_menu() {
    while true; do
        clear
        echo "Choose an option:"
        echo "1. Full installation"
        echo "2. Installation for deployment"
        echo "3. Installation for training"
        echo "4. Installation without creating conda env"
        echo "5. Custom installation"
        echo "9. Back"
        echo "0. Exit"

        read -p "Enter your choice: " choice
        echo    # move to a new line

        case $choice in
            1)
                installation 0 1 2 3 4 5 false
                break
                ;;
            2)
                installation 0 1 2 3 5 false
                break
                ;;
            3)
                installation 0 1 2 4 5 false
                break
                ;;
            4)
                installation 0 1 2 5 false
                break
                ;;
            5)
                installation 0 1 2 3 4 5 true
                break
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

# Start the main menu
main_menu
