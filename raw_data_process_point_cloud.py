from tqdm import tqdm
import json, os, zarr
import numpy as np
from observation.depth_image_process import process_depth_image_offline
from PIL import Image


def read_json(json_file, data_dict, image_params):
    with open(json_file, "r") as file:
        data = json.load(file)

    data_path = os.path.dirname(json_file)

    for item in data:
        rgb_img_file = os.path.join(
            data_path, item["rgb"].split(data_path.split("/")[-1] + "/")[1]
        )
        depth_image_file = os.path.join(
            data_path, item["depth"].split(data_path.split("/")[-1] + "/")[1]
        )
        rgb_img = Image.fromarray(np.load(rgb_img_file))
        assert rgb_img.size == (960, 540)
        rgb_img = rgb_img.crop((230, 0, 960, 540))
        depth_img = Image.fromarray(np.load(depth_image_file))
        assert depth_img.size == (960, 540)
        depth_img = depth_img.crop((230, 0, 960, 540))
        point_cloud = process_depth_image_offline(
            np.array(rgb_img), np.array(depth_img)
        )

        data_dict["colored_clouds"].append(point_cloud)
        data_dict["states"].append(item["pose"])
        data_dict["actions"].append(item["pose"])
    data_dict["episode_ends"].append(len(data_dict["colored_clouds"]))


def process(save_path, json_files, image_params):
    colored_clouds = []
    actions = []
    states = []
    imgs = []
    wrist_imgs = []
    scene_imgs = []
    episode_ends = []
    data = {
        "colored_clouds": colored_clouds,
        "actions": actions,
        "states": states,
        "imgs": imgs,
        "wrist_imgs": wrist_imgs,
        "scene_imgs": scene_imgs,
        "episode_ends": episode_ends,
    }

    for json_file in tqdm(json_files):
        read_json(json_file, data, image_params)

    colored_clouds = np.array(colored_clouds).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    states = np.array(states).astype(np.float32)

    episode_ends = np.array(episode_ends).astype(np.int64)

    print(colored_clouds.shape, actions.shape, states.shape, episode_ends)
    # exit()
    with zarr.open(save_path, mode="w") as zf:
        data_group = zf.create_group("data")
        data_group.create_dataset("action", data=actions, dtype="float32")
        data_group.create_dataset("point_cloud", data=colored_clouds, dtype="float32")
        data_group.create_dataset("state", data=states, dtype="float32")

        data_group = zf.create_group("meta")
        data_group.create_dataset("episode_ends", data=episode_ends, dtype="int64")


def run(data_path_list, save_path, image_params):

    for data_path in data_path_list:
        folders_name = os.listdir(data_path)
        json_files = [
            os.path.join(data_path, folder_name, "data.json")
            for folder_name in folders_name
        ]
        process(save_path, json_files, image_params)


def run_folder_list(json_folder_list, save_path):

    json_files = [
        os.path.join(folder_name, "data.json") for folder_name in json_folder_list
    ]
    process(save_path, json_files)


def run_512():
    params_512_512 = {
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

    data_path_list = [
        # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1211",
        "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1219",
    ]
    save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1219"
    run(data_path_list, save_path, params_512_512)

    # data_path_list = [
    #     "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213/cube_a29",
    #     "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213/cube_a17",
    # ]
    # save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1213_a29_a17"
    # run_folder_list(data_path_list, save_path)

    print("done")


def run_240():

    data_path_list = [
        "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1224",
        # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1219",
    ]
    save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/pc_crop_1224"
    run(data_path_list, save_path, None)

    print("done")


if __name__ == "__main__":
    run_240()
