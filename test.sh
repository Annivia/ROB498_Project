#!/bin/bash

# Change to the TrainA folder
cd checkpoints/TrainA

# Use the find command to search for subsubdirectories with names matching the pattern 'checkpoint00000[1-9]'
# -mindepth 2 and -maxdepth 2 ensure that only the subsubdirectories are searched
# -type d specifies that we are looking for directories
# -regex is used to match the required pattern

find . -mindepth 2 -maxdepth 2 -type d | while read -r dir; do
    # Echo the directory name of the subsubfolder
    python ../../train.py --test --checkpoint_dir=$(realpath "$dir")
done

