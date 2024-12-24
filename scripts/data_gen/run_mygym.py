"""
Script running experiments in the simulated robotic environment myGym
"""


import os
import pickle
import random

import numpy as np
import numpy.random as npr
import pybullet as p
import pybullet_data
import ray
from ray.experimental.tqdm_ray import tqdm

from myGym.train import configure_env, get_parser, get_arguments


@ray.remote
def mental_sim(arg_dict: dict[str, ...]) -> None:
    """
    Runs data collection for the mental simulation in Experiment 1

    :param arg_dict: Arguments for the environment
    """

    arg_dict['seed'] = npr.randint(10, 10_000)

    # Basic configuration required by the myGym simulator
    env = configure_env(arg_dict, os.path.dirname(arg_dict.get('model_path', '')), for_train=0)
    env.render("human")

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    data = {}

    try:
        # Initialize the environment
        env.reset()

        # Generate trajectories of length `n_steps`
        for n_steps in range(1, 11):
            data[n_steps] = []

            # Run trials
            for _ in tqdm(range(10_000), desc=f'n_steps = {n_steps}'):
                series = []

                # Generate a single trajectory
                for _ in range(n_steps + 1):
                    action = env.action_space.sample()
                    observation, *_ = env.step(action)

                    # Observation is current_obj_xyz (3D), goal_obj_xyz (3D), arm_joint_config (7D), end_eff_xyz (3D)
                    # The object is irrelevant in this experiment -> we observe only arm_joint_config, end_eff_xyz
                    series.append(observation[6:])

                data[n_steps].append(series)

    finally:
        print('Saving observations...')

        with open(f'./train_data_{arg_dict["seed"]}.pkl', 'wb') as f:
            pickle.dump(data, f)


@ray.remote
def motor_babbling(arg_dict: dict[str, ...], record: bool = False) -> None:
    """
    Motor babbling for Experiment 1

    :param arg_dict: Arguments for the environment
    :param record: If True, captures frames for visualization purposes
    """

    arg_dict['seed'] = npr.randint(10, 10_000)

    # Basic configuration required by the myGym simulator
    env = configure_env(arg_dict, os.path.dirname(arg_dict.get('model_path', '')), for_train=0)
    env.render("human")

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # We want to directly control joints of the arm
    assert 'joints' in arg_dict['robot_action']
    assert env.episode_max_time > 10_000

    data = []
    images = []

    try:
        # Initialize the environment
        obs = env.reset()
        data.append(obs)

        for t in range(arg_dict["max_episode_steps"]):
            action = env.action_space.sample()

            obs, *_ = env.step(action)
            data.append(obs)

            if record:
                render_info = env.render(mode='rgb_array', camera_id=0)
                image = render_info[0]['image']
                images.append(image)

    finally:
        print('Saving observations...')

        with open(f'./train_data_{arg_dict["seed"]}.pkl', 'wb') as f:
            pickle.dump(data, f)

        if record:
            with open(f'./images_{arg_dict["seed"]}.pkl', 'wb') as f:
                pickle.dump(images, f)


@ray.remote
def general_model_magnet(arg_dict: dict[str, ...], record: bool = False) -> None:
    """
    Data collection for Experiment 2

    :param arg_dict: Arguments for the environment
    :param record: If True, captures frames for visualization purposes
    """

    def box_sample(xyz):
        x_lim = (-0.7, 0.7)
        y_lim = (0.2, 0.85)
        z_lim = (0.1, 0.6)

        min_lim, max_lim = zip(*[x_lim, y_lim, z_lim])

        return np.clip(
            a=xyz + npr.randn(3) * 0.2,
            a_min=min_lim,
            a_max=max_lim
        ) - xyz

    arg_dict['seed'] = npr.randint(10, 10_000)

    # Basic configuration required by the myGym simulator
    env = configure_env(arg_dict, os.path.dirname(arg_dict.get('model_path', '')), for_train=0)
    env.render("human")

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.2, 180, -30, [0.0, 0.5, 0.05])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Action is a displacement of the end effector
    assert 'step' in arg_dict['robot_action']

    data = []
    images = []

    try:
        for ep in tqdm(range(1000), desc=f'Episode'):
            # Randomize color, a control variable
            color = npr.rand(3)
            ep_data = []

            # Initialize the environment
            obs = env.reset()
            (
                obj_xyz, obj_rot, _,
                joints, eff_xyz, eff_rot
            ) = np.split(obs, [3, 7, 14, 21, 24])

            obs_counter = 0
            obj_moves = False
            obj_0 = obj_xyz

            env.env.robot.set_magnetization(True)
            magnet_t = 0
            cooldown_t = 0

            for t in range(arg_dict["max_episode_steps"]):
                # If the object fell off the table, reset
                if obj_xyz[2] < -0.01:
                    break

                # Hardcoded policy facilitating exploration of the state-action space
                # After the cooldown period, reactivate the magnet
                if cooldown_t <= 0:
                    env.env.robot.magnetize_object(env.env.env_objects['actual_state'])

                if env.env.robot.gripper_active:
                    # If the object is attached to the magnetic endpoint

                    # If the manoeuvring period has ended
                    if magnet_t <= 0:
                        # Release the object and start the cooldown period
                        env.env.robot.release_all_objects()
                        magnet_t = random.randint(10, 100)
                        cooldown_t = random.randint(5, 20)
                    else:
                        # Manoeuvre with it in the air
                        dist_vec = box_sample(eff_xyz)
                        magnet_t -= 1
                elif cooldown_t > 0:
                    # If the object is not attached and the cooldown is in progress
                    # Babble around empty-handed
                    dist_vec = box_sample(eff_xyz)
                    cooldown_t -= 1
                else:
                    # If the cooldown has ended, navigate to the object ASAP
                    dist_vec = obj_xyz - eff_xyz

                action = dist_vec
                obs, *_ = env.step(action)

                if record:
                    render_info = env.render(mode='rgb_array', camera_id=3)
                    image = render_info[3]['image']
                    images.append(image)

                (
                    obj_xyz, obj_rot, _,
                    joints, eff_xyz, eff_rot
                ) = np.split(obs, [3, 7, 14, 21, 24])

                # Only record observations when the object is attached to the magnet
                # or when the object is static, i.e. it settles down in a stable position
                if env.env.robot.gripper_active or not obj_moves:
                    obs_counter += 1

                    obs = np.concatenate([
                        obj_xyz, obj_rot, color,
                        joints, eff_xyz, eff_rot,
                        [int(env.env.robot.gripper_active)]
                    ])

                    ep_data.append(obs)

                obj_moves = not np.allclose(obj_0, obj_xyz, rtol=0.0, atol=1e-3)
                obj_0 = obj_xyz

            data.append(ep_data)

    finally:
        print('Saving observations...')

        with open(f'./train_data_{arg_dict["seed"]}.pkl', 'wb') as f:
            pickle.dump(data, f)

        if record:
            with open(f'./images_{arg_dict["seed"]}.pkl', 'wb') as f:
                pickle.dump(images, f)


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument("-ct", "--control", default="slider",
                        help="How to control robot during testing. Valid arguments: keyboard, observation, random, oraculum, slider")
    parser.add_argument("-vs", "--vsampling", action="store_true", help="Visualize sampling area.")
    parser.add_argument("-vt", "--vtrajectory", action="store_true", help="Visualize gripper trajectgory.")
    parser.add_argument("-vn", "--vinfo", action="store_true", help="Visualize info. Valid arguments: True, False")
    parser.add_argument("-nl", "--natural_language", default=False, help="NL Valid arguments: True, False")

    args = get_arguments(parser)
    args_ref = ray.put(args)

    n_processes = 1
    # experiment = mental_sim
    experiment = motor_babbling
    # experiment = general_model_magnet

    ray.get([
        experiment.remote(args_ref)
        for _ in range(n_processes)
    ])
