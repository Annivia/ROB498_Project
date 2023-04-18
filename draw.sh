# #!/bin/bash
#
# cd checkpoints
#
# # Use the find command to search for subsubdirectories with names matching the pattern 'checkpoint00000[1-9]'
# # -mindepth 2 and -maxdepth 2 ensure that only the subsubdirectories are searched
# # -type d specifies that we are looking for directories
# # -regex is used to match the required pattern
#
# find . -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
#     # Echo the directory name of the subsubfolder
#     python ../draw_trajectories.py --dir=$(realpath "$dir/results.json")
# done

#!/bin/bash

# Set the absolute path to the 'checkpoints' folder
checkpoints_dir="checkpoints"

# Use the find command to search for subdirectories with names matching the pattern 'checkpoint00000[1-9]'
# -mindepth 1 and -maxdepth 1 ensure that only the subdirectories are searched
# -type d specifies that we are looking for directories
# -regex is used to match the required pattern

find "$checkpoints_dir" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    # Echo the directory name of the subfolder
    python ./draw_trajectories.py --dir=$(realpath "$dir/results.json")
done
