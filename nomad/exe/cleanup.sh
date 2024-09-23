#!/usr/bin/env bash

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

# Function to remove conda environment
remove_conda_env() {
    local env_name=$1
    source /opt/miniconda3/etc/profile.d/conda.sh
    conda deactivate
    conda env remove -n "$env_name" -y
    conda clean --all -y
}

# Array of cleanup tasks
r="rm -rf src/nomad/"
tasks=(
    "Delete rosbags|${r}preprocessing/rosbags"
    "Delete topomaps|${r}preprocessing/topomap"
    "Delete training data|${r}preprocessing/training_data"
    "Delete data splits|${r}preprocessing/data_splits"
    "Delete wandb cache|${r}train/wandb"
    "Delete train logs|${r}train/logs"
    "Delete train config|${r}train/config/path.yaml"
    "Delete deploy config|${r}deploy/config/navigate.yaml"
    "Delete model weights|${r}deploy/model_weights"
    "Remove diffusion_policy|cd src/ && git submodule deinit -f --all > /dev/null 2>&1 && cd .."
    "Unbuild colcon|rm -rf build log install"
    "Uninstall train|find src/nomad/train -type d -name '*egg-info*' -exec rm -r {} +"
    "Delete pycache|find . -type d -name '__pycache__' -exec rm -r {} +"
    "Remove deploy env|remove_conda_env deploy_nomad"
    "Remove train env|remove_conda_env train_nomad"
)

# Function to execute a range of tasks
execute_tasks() {
    local start_index=$1
    local end_index=$2
    local confirm_tasks=$3

    for i in $(seq $start_index $end_index); do
        IFS="|" read -r task_name command <<< "${tasks[$i]}"
        if [ "$confirm_tasks" = true ]; then
            confirm_and_execute "$task_name?" "$command" "$task_name"
        else
            eval "$command"
            echo "$task_name completed."
        fi
    done
}

# Main menu function
main_menu() {
    while true; do
    	clear
        echo "Choose an option:"
        echo "1. Delete collected data"
        echo "2. Clear training logs"
        echo "3. Custom removal"
        echo "4. Clean repository"
        echo "5. Uninstall"
        echo "9. Back"
        echo "0. Exit"

        read -p "Enter your choice: " choice
        echo    # move to a new line

        case $choice in
            1)
                execute_tasks 0 2 false
                break
                ;;
            2)
                execute_tasks 3 5 false
                break
                ;;
            3)
                execute_tasks 0 14 true
                break
                ;;
            4)
                execute_tasks 0 12 false
                break
                ;;
            5)
                execute_tasks 0 14 false
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

