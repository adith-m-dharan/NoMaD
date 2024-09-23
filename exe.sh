#!/usr/bin/env bash

while true; do
    clear
    echo "Please choose an option:"
    echo "1. Setup"
    echo "2. Run"
    echo "3. Tune"
    echo "4. Train"
    echo "5. Cleanup"
    echo "0. Exit"
    read -p "Enter your choice: " choice

    case $choice in
        1)
            ./src/nomad/exe/setup.sh
            break
            ;;
        2)
            ./src/nomad/exe/run.sh
            break
            ;;
        3)
            ./src/nomad/exe/tune.sh
            break
            ;;
        4)
            ./src/nomad/exe/train.sh
            break
            ;;
        5)
            ./src/nomad/exe/cleanup.sh
            break
            ;;
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
