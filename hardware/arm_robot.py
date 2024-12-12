from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
import sys
sys.path.insert(0,"/home/robot/UR_Robot_Arm_Show/tele_ws/src/tele_ctrl_jeff/scripts")
from robotiq_gripper import RobotiqGripper
import time
import numpy as np
from scipy.spatial.transform import Rotation as R


def axis_to_euler(axis_angle):
    """
    @input : rx,ry,rz - [axis angle] | array
    @output : rx,ty,tz - [euler] | array
    """
    theta = np.linalg.norm(axis_angle)
    axis = axis_angle / theta
    half_theta = theta / 2
    q_w = np.cos(half_theta)
    q_xyz = axis * np.sin(half_theta)
    quat = np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w])  # [x,y,z,w]
    rotation = R.from_quat(quat)  # [x,y,z,w]
    euler = rotation.as_euler("xyz", degrees=False)
    return euler


def axis_to_quat(axis_angle):
    """
    @input : rx,ry,rz - [axis angle] | array
    @output : qx,qy,qz,qw - [quat] | array
    """
    theta = np.linalg.norm(axis_angle)
    axis = axis_angle / theta
    half_theta = theta / 2
    q_w = np.cos(half_theta)
    q_xyz = axis * np.sin(half_theta)
    quat = np.array([q_xyz[0], q_xyz[1], q_xyz[2], q_w])  # [x,y,z,w]
    return quat


def euler_to_axis(euler_angles):
    """
    Converts Euler angles to axis-angle representation.
    @input : rx, ry, rz - [euler angles] | array
    @output : rx, ry, rz - [axis angle] | array
    """
    rotation = R.from_euler("xyz", euler_angles, degrees=False)
    axis_angle = rotation.as_rotvec()
    return axis_angle


def quat_to_axis(quaternion):
    """
    Converts quaternion to axis-angle representation.
    @input : qx, qy, qz, qw - [quaternion] | array
    @output : rx, ry, rz - [axis angle] | array
    """
    quaternion = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    rotation = R.from_quat(quaternion)
    axis_angle = rotation.as_rotvec()
    return axis_angle


class ArmRobot:

    def __init__(self) -> None:
        self.ROBOT_HOST = "192.168.1.17"
        self.MOVE_SPEED = 0.15
        self.GP_OPEN = 0
        self.GP_CLOSE = 1
        self.GP_CRITERIA = 0.5
        self.IS_VERBOSE = True
        self.IS_CHECK = True
        self.IS_SAVE = True
        self.IS_CARP = True
        self.gripper_status = self.GP_OPEN
        self.rtde_ctl = RTDEControlInterface(self.ROBOT_HOST)
        self.rtde_rcv = RTDEReceiveInterface(self.ROBOT_HOST)
        self.gripper = RobotiqGripper()
        print("[INFO] Connecting to gripper...")
        self.gripper.connect(self.ROBOT_HOST, 63352)
        print("[INFO] Activating gripper...")
        self.gripper.activate()

    def init_action(
        self,
        action=[
            -0.01080731052415074,
            -0.5574208651490298,
            0.372392988522742,
            0.16757318980069139,
            0.9858072989365493,
            -0.001075519806611833,
            0.010101419731514072,
            0.0,
        ],
    ):
        next_state = [*action[:3], *quat_to_axis(action[3:7])]
        self.rtde_ctl.moveL(next_state, speed=self.MOVE_SPEED)
        if action[-1] < self.GP_CRITERIA and self.gripper_status == self.GP_CLOSE:
            self.gripper_status = self.GP_OPEN
            self.gripper.move(self.gripper.get_open_position(), 255, 255)

        print(f"the robot is initing motion, please waiting for this process over...")
        time.sleep(5)
        print(f"the robot init motion over.")

    def get_state(self):
        actual_pose = self.rtde_rcv.getActualTCPPose()
        axis_cart = np.array(actual_pose[:3])
        axis_angle = np.array(actual_pose[3:])
        euler = axis_to_euler(axis_angle)
        quat = axis_to_quat(axis_angle)

        data = np.array(
            axis_cart.tolist() + quat.tolist() + list([float(self.gripper_status)])
        )

        return data

    def run(self, actions_pred):
        actions_pred = actions_pred.reshape(-1, 8)  # [H,D] | pos(3) + rot(4) + grip(1)
        actions_pred = actions_pred[::2, ...]

        # with open('/media/robot/2CCF4D6BBC2D923E/tt/1206/cube3/data.json', 'r') as file:
        #     actions_pred = json.load(file)

        ### run
        for i, action in enumerate(actions_pred):
            # action = np.array(action['pose'])
            ## move
            next_state = [*action[:3], *quat_to_axis(action[3:7])]
            self.rtde_ctl.moveL(next_state, speed=self.MOVE_SPEED)

            ## gripper
            if action[-1] > self.GP_CRITERIA and self.gripper_status == self.GP_OPEN:
                self.gripper_status = self.GP_CLOSE
                self.gripper.move(self.gripper.get_closed_position(), 255, 255)
            if action[-1] < self.GP_CRITERIA and self.gripper_status == self.GP_CLOSE:
                self.gripper_status = self.GP_OPEN
                self.gripper.move(self.gripper.get_open_position(), 255, 255)
            if i == 0:
                time.sleep(1)
            else:
                time.sleep(0.05)
