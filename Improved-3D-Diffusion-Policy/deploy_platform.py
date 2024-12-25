import sys
import time
import torch
import numpy as np
import os

import hydra

os.environ["WANDB_SILENT"] = "True"

sys.path.append("/home/robot/UR_Robot_Arm_Show/tele_ws/src/tele_ctrl_jeff/scripts/")
from inference import InferenceEnv

sys.path.append("/home/robot/ArmRobot")
from observation.Image_process_utils import process_image_npy
from hardware.arm_robot import ArmRobot




class Inference:
    """
    The deployment is running on the local computer of the robot.
    """

    def __init__(
        self,
        robot: ArmRobot,
        obs_horizon=2,
        action_horizon=8,
        device="gpu",
    ):
        self.robot = robot

        # camera
        args = {"fps": 20, "visualize": False, "path": ""}
        self.obs_env = InferenceEnv(args)
        print("self.obs_env: ", self.obs_env)

        # horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # inference device
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.image_params = {
            "rgb": {
                "size": (960, 540),
                "crop": (230, 0, 770, 540),
                "resize": (512, 512),
            },
            "wrist": {
                "size": (640, 480),
                "crop": (0, 0, 640, 480),
                "resize": (512, 512),
            },
            "scene": {
                "size": (640, 480),
                "crop": (100, 160, 540, 480),
                "resize": (512, 512),
            },
        }

    def reset_robot(self):

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
        self.robot.init_action(action)

        self.obs_list = []

    def construct_obs(self, obs_list):
        """
        从数据采集框架中拿到数据进行处理
        Note: 与预训练时处理流程要保持一样
        """

        if len(obs_list) < self.obs_horizon:
            env_obs = [obs_list[0]] * self.obs_horizon
        else:
            env_obs = obs_list[-self.obs_horizon :]

        wrist_image_list = []
        rgb_image_list = []
        state_list = []
        for obs in env_obs:

            wrist_image = process_image_npy(obs["wrist"], "wrist", self.image_params)
            wrist_image = np.transpose(wrist_image, (2, 0, 1))
            wrist_image_list.append(wrist_image)
            rgb_image = process_image_npy(obs["rgb"], "rgb", self.image_params)
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
            rgb_image_list.append(rgb_image)

            state_list.append(obs["state"])

        wrist = np.stack(wrist_image_list)
        rgb = np.stack(rgb_image_list)
        state = np.stack(state_list)

        model_input = {}
        model_input["wrist_image"] = (
            torch.from_numpy(wrist).unsqueeze(0).to(self.device)
        )
        model_input["image"] = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        model_input["agent_pos"] = torch.from_numpy(state).unsqueeze(0).to(self.device)

        return model_input


    def filter_actions(self, actions):
        """
        对 policy 输出做调整，比如间隔取值
        """
        return actions["action"].detach().cpu().numpy()[0]

    def step_one(self, action):
        print("step: ", action.shape)
        self.robot.send_action(action)

    def get_obs_dict(self):
        data = self.obs_env.get_state_obs()
        data["state"] = self.robot.get_state()

        return data

    def run(self, policy):
        step_count = 0

        while step_count < 1000:
            with torch.no_grad():

                obs = self.get_obs_dict()
                print(obs.keys())
                self.obs_list.append(obs)

                model_input = self.construct_obs(self.obs_list)

                actions = policy.predict_action(model_input)

            actions = self.filter_actions(actions)

            for action in actions:
                self.step_one(action)
                time.sleep(0.1)
                step_count += 1

            print(f"step: {step_count}")


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

    # fetch policy model
    policy = workspace.get_model()
    
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    print(f"==== action_horizon: {action_horizon} ====")


    """
    run policy
    """
    robot = ArmRobot()

    env = Inference(
        robot=robot,
        obs_horizon=2,
        action_horizon=action_horizon,
        device="cpu",
    )

    env.reset_robot()

    env.run(policy)


def test():

    print("==== test ==")
    robot = ArmRobot()

    env = Inference(
        robot=robot,
        obs_horizon=2,
        action_horizon=16,
        device="cpu",
    )

    env.reset_robot()

    env.run(None)


if __name__ == "__main__":
    main()
    # test()
