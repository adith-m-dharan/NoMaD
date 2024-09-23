#!/bin/bash

# Paths to the YAML files
NAVIGATE_YAML="src/nomad/deploy/config/navigate.yaml"
MODEL_YAML="src/nomad/train/config/model.yaml"
CONTROLLER_YAML="src/nomad/deploy/config/controller.yaml"

# Function to modify a variable in the YAML file
update() {
    local file=$1
    local var_name=$2
    local new_value=$3
    sed -i "s/^\(\s*$var_name:\).*/\1 $new_value/" "$file"
}

# Function to read the current value of a variable from the YAML file
show() {
    local file=$1
    local var_name=$2
    grep "^\s*$var_name:" "$file" | sed "s/^\s*$var_name:[[:space:]]*//"
}

# Function to handle input and variable modification
tune() {
    local var_name=$1
    local file=$2
    local current_value=$(show "$file" "$var_name")
    read -p "Enter new value for $var_name: " new_value
    update "$file" "$var_name" "$new_value"
}

# Main menu function
main_menu() {
    while true; do
        clear
        echo "Choose a variable to tune:"
        echo "A. Goal image (current: $(show "$NAVIGATE_YAML" "goal_node"))"
        echo "B. Tolerence in the goal image (current: $(show "$NAVIGATE_YAML" "tolerence"))"
        echo "C. Radius images to be compared (current: $(show "$NAVIGATE_YAML" "radius"))"
        echo "D. Skip in the radius image (current: $(show "$NAVIGATE_YAML" "skip"))"
        echo "E. Maximum linear velocity of the model (current: $(show "$NAVIGATE_YAML" "v_max"))"
        echo "F. Maximum angular velocity of the model (current: $(show "$NAVIGATE_YAML" "w_max"))"
        echo "G. Controller waypoint timeout (current: $(show "$CONTROLLER_YAML" "waypoint_timeout"))"
        echo "H. Maximum linear velocity of the robot (current: $(show "$CONTROLLER_YAML" "v_max"))"
        echo "I. Maximum angular velocity of the robot (current: $(show "$CONTROLLER_YAML" "w_max"))"
        echo "J. Number of diffusion iterations (current: $(show "$MODEL_YAML" "num_diffusion_iters"))"
        echo "K. Length of future predictions (current: $(show "$MODEL_YAML" "len_traj_pred"))"
        echo "L. Images from topic to be compared (current: $(show "$MODEL_YAML" "context_size"))"
        echo "M. Controller loop frequency (current: $(show "$CONTROLLER_YAML" "frame_rate"))"
        echo "N. Model loop frequency (current: $(show "$NAVIGATE_YAML" "hz"))"
        echo "O. Graph frequency (current: $(show "$NAVIGATE_YAML" "graph_hz"))"
        echo "P. Number of samples (current: $(show "$NAVIGATE_YAML" "num_samples"))"
        echo "Q. Waypoint (current: $(show "$NAVIGATE_YAML" "waypoint"))"
        echo "R. Close threshold (current: $(show "$NAVIGATE_YAML" "close_threshold"))"
        echo "Z. Back"
        echo "X. Exit"
        read -p "Enter your choice: " choice
        echo    # move to a new line

        case ${choice^^} in
	    A) tune "goal_node" "$NAVIGATE_YAML" ;;
	    B) tune "tolerence" "$NAVIGATE_YAML" ;;
	    C) tune "radius" "$NAVIGATE_YAML" ;;
	    D) tune "skip" "$NAVIGATE_YAML" ;;
	    E) tune "v_max" "$NAVIGATE_YAML" ;;
	    F) tune "w_max" "$NAVIGATE_YAML" ;;
	    G) tune "waypoint_timeout" "$CONTROLLER_YAML" ;;
	    H) tune "v_max" "$CONTROLLER_YAML" ;;
	    I) tune "w_max" "$CONTROLLER_YAML" ;;
	    J) tune "num_diffusion_iters" "$MODEL_YAML" ;;
	    K) tune "len_traj_pred" "$MODEL_YAML" ;;
	    L) tune "context_size" "$MODEL_YAML" ;;
	    M) tune "frame_rate" "$CONTROLLER_YAML" ;;
	    N) tune "hz" "$NAVIGATE_YAML" ;;
	    O) tune "graph_hz" "$NAVIGATE_YAML" ;;
	    P) tune "num_samples" "$NAVIGATE_YAML" ;;
	    Q) tune "waypoint" "$NAVIGATE_YAML" ;;
	    R) tune "close_threshold" "$NAVIGATE_YAML" ;;

            Z)
                ./src/exe.sh
                break
                ;;
            X)
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
