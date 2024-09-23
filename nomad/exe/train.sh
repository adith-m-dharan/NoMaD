#!/usr/bin/env bash

source /opt/miniconda3/etc/profile.d/conda.sh

# Change directory to the script's directory
cd "$(dirname "$0")/../train" || exit 1

mkdir -p logs/nomad_run

# Function for training from scratch
train_from_scratch() {
    grep -q "# load_run" config/model.yaml || sed -i '/load_run/s/^/# /' config/model.yaml
    echo "Training from scratch."
}

# Function for training from nomad checkpoint
train_from_nomad_checkpoint() {
    cp ../deploy/model_weights/nomad.pth logs/nomad_run/latest.pth
    grep -q "# load_run" config/model.yaml && sed -i '/# load_run/s/^# //' config/model.yaml
    echo "Training with nomad checkpoint."
}

# Function for training from previous checkpoint
train_from_previous_checkpoint() {
    echo "Manually paste your checkpoint in nomad_run folder."
    read -p "Press Enter to continue..."
    mv logs/nomad_run/*.pth logs/nomad_run/latest.pth
    grep -q "# load_run" config/model.yaml && sed -i '/# load_run/s/^# //' config/model.yaml
    echo "Training with previous checkpoint."
}

# Function to prompt user for training mode and handle invalid options
training_menu() {
    while true; do
        clear
        echo "Choose training mode:"
        echo "1. Train from scratch"
        echo "2. Train from nomad checkpoint"
        echo "3. Train from previous checkpoint"
#        echo "9. Back"
        echo "0. Exit"
        read -p "Enter your choice (0-3): " choice
        echo    # move to a new line

        case $choice in
            1)
                train_from_scratch
                break
                ;;
            2)
                train_from_nomad_checkpoint
                break
                ;;
            3)
                train_from_previous_checkpoint
                break
                ;;
#            9)
#                ./src/exe.sh
#                break
#                ;;
            0)
                echo "Exiting."
                sleep 1
                clear
                exit 0
                ;;
            *)
                echo "Invalid option. Please try again."
                sleep 2
                ;;
        esac
    done
}

# Function to activate conda environment and run training scripts
run_training() {
    conda activate train_nomad || exit 1
    python3 data_split.py
    python3 train.py
}

# Main function to start the training menu and run training
main() {
    training_menu
    run_training
}

# Start the main function
main
