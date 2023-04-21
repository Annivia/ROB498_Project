## Reinforcement Learning for Multi-object Manipulation in Cluttered Environments: A Comparative Study

This code represents the final project for the ROB 498 course at the University of Michigan. Some sections of the code have been adapted from the resources provided by the course.

Author: Pony Zhang



### Environment

Run `./install.sh` to install the proper environment. 

Note that the script requires strictly Python = 3.10. If the script fails to resolve your operating system / package manager, please install Python 3.10 manually.



### Usage

For short recap of the project, run

`python demo.py`

It shall display a robot arm completing a planar pushing task. 



For the entire training, run

`./train.sh`

`./test.sh`

to train and test the models. In addition, run

`./draw.sh`

to visualize performance of the trained models.



You could also choose to run `./run_before_sleep.sh`, which is a combination of the three.



### Comments

You may notice lots of redundancy in the four gym environment files starting with *panda_pushing_env*. However, the author of this project didn't figure out a way to pass env_config to the RLlib framework. As a result, she had to continue with four duplicate files of different reward functions. If you know a way to fix this issue, please let her know.
