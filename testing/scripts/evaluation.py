from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from stable_baselines3.common import base_class
import supersuit.vector.sb3_vector_wrapper as sb3vw


def evaluate_policy(
    # model: "base_class.BaseAlgorithm",
    # # env: Union[sb3vw.SB3VecEnvWrapper, VecEnv],
    # num_robots: int,
    # env: sb3vw.SB3VecEnvWrapper,
    robots: Dict[str, Dict[str, Any]],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    obs = {robot: {} for robot in robots}

    not_reseted = True
    # Avoid double reset, as VecEnv are reset automatically.
    if not_reseted:
        for robot in robots:
            # ('robot': {'agent1': obs, 'agent2': obs, ...})
            obs[robot] = robots[robot]["env"].reset()
            not_reseted = False

    # {'robot': [agents]} e.g. {'jackal': [agent1, agent2], 'burger': [agent1]}
    agents = {robot: robots[robot]["agent"] for robot in robots}

    # {
    #   'robot1':
    #       'agent1': [reward1, reward2, ...]
    #       'agent2': [reward1, reward2, ...]
    #   'robot2':
    #       'agent1': [reward1, reward2, ...]
    #       'agent2': [reward1, reward2, ...]
    # }
    for robot in robots:
        episode_rewards = {robot: {agent: []} for agent in agents[robot]}
    episode_lengths = []

    # dones -> {'robot': {'agent': False}}
    # e.g. {'jackal': {agent1: False, agent2: False}, 'burger': {'agent1': False}}
    default_dones = {robot: {a: None for a in agents[robot]} for robot in robots}
    default_states = {}
    default_actions = {}
    # states, actions, episode rewards -> {'robot': {'agent': None}}
    # e.g. {'jackal': {agent1: None, agent2: None}, 'burger': {'agent1': None}}
    default_episode_reward = {
        robot: {a: None for a in agents[robot]} for robot in robots
    }
    while len(episode_rewards) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not_reseted:
            for robot in robots:
                # ('robot': {'agent1': obs, 'agent2': obs, ...})
                obs[robot] = robots[robot]["env"].reset()
                not_reseted = False

        dones = default_dones.copy()
        states = default_states.copy()
        actions = default_actions.copy()
        episode_reward = default_episode_reward.copy()
        episode_length = 0
        while not check_dones(dones):
            # Go over all robots
            for robot in robots:
                # Get predicted actions and new states from each agent
                for agent, state in states[robot].items():
                    actions[agent], states[agent] = robots[robot]["model"].predict(
                        obs[robot][agent], state, deterministic=deterministic
                    )

                # Publish actions in the environment
                robots[robot]["env"].apply_actions(actions)
                # And get new obs, rewards, dones, and infos
                obs[robot], rewards, dones[robot], _ = robots[robot]["env"].get_states()
                # Add up rewards for this episode
                for agent, reward in zip(agents[robot], rewards.values()):
                    episode_reward[robot][agent] += reward

                if callback is not None:
                    callback(locals(), globals())

                if render:
                    robots[robot]["env"].render()

            # Take one step in the simulation (applies to all robots and all agents, respectively!)
            robots[robot]["env"].call_service_takeSimStep()

            episode_length += 1

        # if is_monitor_wrapped:
        #     # Do not trust "done" with episode endings.
        #     # Remove vecenv stacking (if any)
        #     # if isinstance(env, VecEnv):
        #     #     info = info[0]
        #     if "episode" in info.keys():
        #         # Monitor wrapper includes "episode" key in info if environment
        #         # has been wrapped with it. Use those rewards instead.
        #         episode_rewards.append(info["episode"]["r"])
        #         episode_lengths.append(info["episode"]["l"])
        # else:

        # For each robot append rewards for every of its agents
        # to the list of respective episode rewards
        for robot in robots:
            for agent, reward in episode_reward[robot].items():
                episode_rewards[robot][agent].append(reward)

        episode_lengths.append(episode_length)

    # CURRENTLY HERE IN DEVELOPMENT
    mean_rewards = {agent: np.mean(episode_rewards[agent]) for agent in agents}
    std_rewards = {agent: np.std(episode_rewards[agent]) for agent in agents}
    if reward_threshold is not None:
        assert min(mean_rewards.values()) > reward_threshold, (
            "Atleast one mean reward below threshold: "
            f"{min(mean_rewards.values()):.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_rewards, std_rewards


def check_dones(dones):
    # Check if all agents for every robot are done
    for robot in dones:
        for agent in dones[robot]:
            if dones[robot][agent]:
                continue
            else:
                return False
    return True
