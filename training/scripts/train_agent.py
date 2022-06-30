import os
import sys
import time
from datetime import time
from multiprocessing import cpu_count, set_start_method
from typing import Callable, List

import rospkg
import rospy
from rl_utils.rl_utils.envs.pettingzoo_env import env_fn
from rl_utils.rl_utils.training_agent_wrapper import TrainingDRLAgent
from rl_utils.rl_utils.utils.supersuit_utils import vec_env_create
from rl_utils.rl_utils.utils.utils import instantiate_train_drl_agents
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import BaseAgent
from rosnav.model.custom_policy import *
from rosnav.model.custom_sb3_policy import *
from stable_baselines3.common.callbacks import (
    MarlEvalCallback,
    StopTrainingOnRewardThreshold,
)
from task_generator.robot_manager import init_robot_managers
from task_generator.tasks import get_MARL_task, get_task_manager, init_obstacle_manager
from training.tools import train_agent_utils
from training.tools.argsparser import parse_training_args
from training.tools.custom_mlp_utils import *

# from tools.argsparser import parse_training_args
from training.tools.staged_train_callback import InitiateNewTrainStage

# from tools.argsparser import parse_marl_training_args
from training.tools.train_agent_utils import *
from training.tools.train_agent_utils import (
    choose_agent_model,
    create_training_setup,
    get_MARL_agent_name_and_start_time,
    get_paths,
    initialize_hyperparameters,
    load_config,
)


def main(args):
    # load configuration
    config = load_config(args.config)
    robot_names = [robot for robot in config["robots"].keys()]

    # set debug_mode
    rospy.set_param("debug_mode", config["debug_mode"])

    # create dicts for all robot types with all necessary parameters,
    # and instances of the respective models and envs
    robots = create_training_setup(config)

    # set num of timesteps to be generated
    n_timesteps = 40000000 if config["n_timesteps"] is None else config["n_timesteps"]

    start = time.time()
    try:
        model = robots[robot_names[0]]["model"]
        model.learn(
            total_timesteps=n_timesteps,
            reset_num_timesteps=True,
            # Übergib einfach das dict für den aktuellen roboter
            # callback=get_evalcallback(
            #     eval_config=config["periodic_eval"],
            #     curriculum_config=config["training_curriculum"],
            #     stop_training_config=config["stop_training"],
            #     train_env=robots[robot_name]["env"],
            #     num_robots=robots[robot_name]["robot_train_params"]["num_robots"],
            #     num_envs=config["n_envs"],
            #     task_mode=config["task_mode"],
            #     PATHS=robots[robot_name]["paths"],
            # ),
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")
    # finally:
    # update the timesteps the model has trained in total
    # update_total_timesteps_json(n_timesteps, PATHS)

    robots[robot_names[0]][model].env.close()
    print(f"Time passed: {time.time() - start}s")
    print("Training script will be terminated")
    sys.exit()


def get_evalcallback(
    eval_config: dict,
    curriculum_config: dict,
    stop_training_config: dict,
    train_env: VecEnv,
    num_robots: int,
    num_envs: int,
    task_mode: str,
    PATHS: dict,
) -> MarlEvalCallback:
    """Function which generates an evaluation callback with an evaluation environment.

    Args:
        train_env (VecEnv): Vectorized training environment
        num_robots (int): Number of robots in the environment
        num_envs (int): Number of parallel spawned environments
        task_mode (str): Task mode for the current experiment
        PATHS (dict): Dictionary which holds hyperparameters for the experiment

    Returns:
        MarlEvalCallback: [description]
    """
    eval_env = env_fn(
        num_agents=num_robots,
        ns="eval_sim",
        agent_list_fn=instantiate_train_drl_agents,
        max_num_moves_per_eps=eval_config["max_num_moves_per_eps"],
        PATHS=PATHS,
    )

    # eval_env = VecNormalize(
    #     eval_env,
    #     training=False,
    #     norm_obs=True,
    #     norm_reward=False,
    #     clip_reward=15,
    #     clip_obs=3.5,
    # )

    return MarlEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        num_robots=num_robots,
        n_eval_episodes=eval_config["n_eval_episodes"],
        eval_freq=eval_config["eval_freq"],
        deterministic=True,
        log_path=PATHS["eval"],
        best_model_save_path=PATHS["model"],
        callback_on_eval_end=InitiateNewTrainStage(
            n_envs=num_envs,
            treshhold_type=curriculum_config["threshold_type"],
            upper_threshold=curriculum_config["upper_threshold"],
            lower_threshold=curriculum_config["lower_threshold"],
            task_mode=task_mode,
            verbose=1,
        ),
        callback_on_new_best=StopTrainingOnRewardThreshold(
            treshhold_type=stop_training_config["threshold_type"],
            threshold=stop_training_config["threshold"],
            verbose=1,
        ),
    )


if __name__ == "__main__":
    set_start_method("fork")
    # args, _ = parse_marl_training_args()
    args, _ = parse_training_args()
    # rospy.init_node("train_env", disable_signals=False, anonymous=True)
    main(args)
