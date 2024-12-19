import sys
import cv2

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
sys.path.insert(0, "/home/robot/ArmRobot")
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
from observation.image_crop import process_image_npy

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

    def construct_obs(self, agent_pos_array, obs_cloud_array, rgb_array):
        agent_pos = np.stack(agent_pos_array, axis=0)
        obs_cloud = np.stack(obs_cloud_array, axis=0)
        image = np.stack(rgb_array, axis=0)

        obs_dict = {
            "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        obs_dict["point_cloud"] = (
            torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        )
        obs_dict['image'] = torch.from_numpy(image).unsqueeze(0).to(self.device)

        print(f"obs_cloud_array, ", obs_cloud_array[0])
        return obs_dict

    def step(self, action):
        print("step: ", action.shape)

        self.robot.run(action)

        action_list = [act for act in action]
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)

            cam_dict = self.camera()
            self.color_array.append(process_image_npy(cam_dict["color"]))
            self.depth_array.append(cam_dict["depth"])
            self.cloud_array.append(cam_dict["point_cloud"])


            self.env_qpos_array.append(self.robot.get_state())

        obs_dict = self.construct_obs(
            self.env_qpos_array[-self.obs_horizon :],
            self.cloud_array[-self.obs_horizon :],
            self.color_array[-self.obs_horizon :]
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
        self.color_array.append(process_image_npy(cam_dict["color"]))
        self.depth_array.append(cam_dict["depth"])
        self.cloud_array.append(cam_dict["point_cloud"])

        # env_qpos = np.zeros((1, 8))
        # env_qpos[0, 6 + 5 + 2 + 5 + 2 : 6 + 5 + 
        self.env_qpos_array.append(self.robot.get_state())

        obs_dict = self.construct_obs(
            [self.env_qpos_array[-1]] * self.obs_horizon,
            [self.cloud_array[-1]] * self.obs_horizon,
            [self.color_array[-1]] * self.obs_horizon,
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

    print(f"==== workspace: {workspace.__class__.__name__} ====")
    # print(f"==== task: {cfg.task} ====")
    if "3D" in workspace.__class__.__name__:
        point_cloud_or_image = True
    else:
        point_cloud_or_image = False
    print(f"==== point_cloud_or_image: {point_cloud_or_image} ====")
    

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

    action = [
        -0.18980697676771513,
        -0.3827291399992533,
        0.31114132703422326,
        -0.417985318626756,
        -0.9083933376685246,
        0.008868446494732254,
        0.005582844140526736,
        0.0,
    ]
    robot.init_action(action)

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
            if point_cloud_or_image:
                del obs_dict["image"]
            else:
                del obs_dict["point_cloud"]
            action = policy(obs_dict)[0]

        obs_dict = env.step(action.numpy())
        step_count += action_horizon
        print(f"step: {step_count}")


if __name__ == "__main__":
    main()
