from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil
from panda_pushing_env import PandaDiskPushingEnv

def create_environment(env_config):
    return PandaDiskPushingEnv(**env_config)


def main ():
    # # init directory in which to save checkpoints
    # chkpt_root = "tmp/exa"
    # shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # # init directory in which to log results
    # ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    # shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # # start Ray -- add `local_mode=True` here for debugging
    # ray.init(ignore_reinit_error=True)

    # # register the custom environment
    # register_env("PandaDiskPushingEnv", create_environment)
    # #register_env(select_env, lambda config: Fail_v1())


    # # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    agent = ppo.PPOTrainer(config, env=PandaDiskPushingEnv)

    # status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    # n_iter = 5

    # # train a policy with RLlib using PPO
    # for n in range(n_iter):
    #     result = agent.train()
    #     chkpt_file = agent.save(chkpt_root)

    #     print(status.format(
    #             n + 1,
    #             result["episode_reward_min"],
    #             result["episode_reward_mean"],
    #             result["episode_reward_max"],
    #             result["episode_len_mean"],
    #             chkpt_file
    #             ))


    # # examine the trained policy
    # policy = agent.get_policy()
    # model = policy.model
    # print(model.base_model.summary())


    chkpt_file='tmp/exa/checkpoint_000005'

    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    # env = gym.make(PandaDiskPushingEnv)
    env = PandaDiskPushingEnv(visualizer=None, render_non_push_motions=True,  camera_heigh=800, camera_width=800, render_every_n_steps=1)

    state = env.reset()
    sum_reward = 0
    n_step = 20

    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        print(state)
        sum_reward += reward

        env.render()

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()