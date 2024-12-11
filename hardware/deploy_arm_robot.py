import sys
import cv2

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

from hardware.arm_robot import ArmRobot
import hydra
import time
from omegaconf import OmegaConf
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import torch
import os

os.environ["WANDB_SILENT"] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

zenoh_path = "/home/gr1p24ap0049/projects/gr1-dex-real/teleop-zenoh"
sys.path.append(zenoh_path)

from observation.real_sense_camera import MultiRealSense

import numpy as np
import torch


class Inference:
    """
    The deployment is running on the local computer of the robot.
    """

    def __init__(
        self,
        robot,
        obs_horizon=2,
        action_horizon=8,
        device="gpu",
    ):
        self.robot = robot

        # camera
        self.camera = MultiRealSense()

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

    def construct_obs(self, agent_pos_array, obs_cloud_array):
        agent_pos = np.stack(agent_pos_array, axis=0)
        obs_cloud = np.stack(obs_cloud_array, axis=0)

        obs_dict = {
            "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        obs_dict["point_cloud"] = (
            torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        )
        # print(f"construct_obs, ", obs_dict["agent_pos"].shape)
        return obs_dict

    def step(self, action):
        print("step: ", action.shape)

        self.robot.run(action[:, 3 + 5 + 5 : 3 + 5 + 5 + 8])

        action_list = [act for act in action]
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)

            cam_dict = self.camera()
            self.color_array.append(cam_dict["color"])
            self.depth_array.append(cam_dict["depth"])
            self.cloud_array.append(cam_dict["point_cloud"])

            # TODO
            env_qpos = np.zeros((1, 32))
            env_qpos[0, 6 + 5 + 2 + 5 + 2 : 6 + 5 + 2 + 5 + 2 + 8] = (
                self.robot.get_state()
            )

            self.env_qpos_array.append(env_qpos[0])

        obs_dict = self.construct_obs(
            self.env_qpos_array[-self.obs_horizon :],
            self.cloud_array[-self.obs_horizon :],
        )

        return obs_dict

    def reset(self, first_init=True):
        print(f"reset: first_init:{first_init}")
        # init buffer
        self.color_array, self.depth_array, self.cloud_array = [], [], []
        self.env_qpos_array = []
        self.action_array = []

        if first_init:
            # ======== INIT ==========
            print("first_init")

        time.sleep(2)

        print("Robot ready!")

        # ======== INIT ==========
        # camera.start()
        cam_dict = self.camera()
        self.color_array.append(cam_dict["color"])
        self.depth_array.append(cam_dict["depth"])
        self.cloud_array.append(cam_dict["point_cloud"])

        cv2.imwrite(
            "/media/robot/2CCF4D6BBC2D923E/mpz/iDP3/Improved-3D-Diffusion-Policy/test.png",
            self.color_array,
        )

        env_qpos = np.zeros((1, 32))
        env_qpos[0, 6 + 5 + 2 + 5 + 2 : 6 + 5 + 2 + 5 + 2 + 8] = self.robot.get_state()
        self.env_qpos_array.append(env_qpos)

        obs_dict = self.construct_obs(
            [self.env_qpos_array[-1][0]] * self.obs_horizon,
            [self.cloud_array[-1]] * self.obs_horizon,
        )

        return obs_dict


@hydra.main(
    config_path=str(
        pathlib.Path(__file__).parent.joinpath("diffusion_policy_3d", "config")
    )
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    assert workspace.__class__.__name__ != "DPWorkspace"

    # fetch policy model
    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    print(f"==== action_horizon: {action_horizon} ====")

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    print(f"==== roll_out_length: {roll_out_length} ====")

    first_init = True

    robot = ArmRobot()

    robot.init_action()

    env = Inference(
        robot=robot,
        obs_horizon=2,
        action_horizon=action_horizon,
        device="cpu",
    )

    obs_dict = env.reset(first_init=first_init)

    step_count = 0

    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]

        obs_dict = env.step(action.numpy())
        step_count += action_horizon
        print(f"step: {step_count}")


if __name__ == "__main__":
    main()
