Monitor the progress of training:

1) Start the RLLib Tensor board by command
tensorboard --logdir=<path to the log directory>

2) Set configuration -- "monitor": True

Questions:
Why is there a TARGET_POSE_OBSTACLES or TARGET_POSE_FREE?
camera_heigh=84 Typo?


if self.done_at_goal:
            if self.include_obstacle:
                at_goal = np.linalg.norm(state[:2] - TARGET_POSE_OBSTACLES[:2]) < 1.2 * DISK_RADIUS
Why is it using DISK_RADIUS?

Visualize 

(RolloutWorker pid=591114) 6
(RolloutWorker pid=591114) get_state start
(RolloutWorker pid=591114) get_object_pos_planar start
(RolloutWorker pid=591114) get_object_pose start
(RolloutWorker pid=591114) get_object_pose end
(RolloutWorker pid=591114) get_object_pos_planar start
(RolloutWorker pid=591114) get_state end
