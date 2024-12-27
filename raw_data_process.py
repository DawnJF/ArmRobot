from tqdm import tqdm
import json, os, zarr
import numpy as np
from observation.Image_process_utils import process_image_npy


def read_json(json_file, data_dict, image_params):
    with open(json_file, "r") as file:
        data = json.load(file)

    data_path = os.path.dirname(json_file)

    for item in tqdm(data):

        rgb_img_file = os.path.join(
            data_path, item["rgb"].split("/" + data_path.split("/")[-1] + "/")[1]
        )
        data_dict["imgs"].append(
            process_image_npy(np.load(rgb_img_file), "rgb", image_params)
        )

        # FIXME
        wrist_img_file = os.path.join(
            data_path,
            rgb_img_file.replace("rgb", "wrist").replace("960x540", "640x480"),
        )
        data_dict["wrist_imgs"].append(
            process_image_npy(np.load(wrist_img_file), "wrist", image_params)
        )
        scene_img_file = os.path.join(
            data_path,
            rgb_img_file.replace("rgb", "scene").replace("960x540", "640x480"),
        )
        data_dict["scene_imgs"].append(
            process_image_npy(np.load(scene_img_file), "scene", image_params)
        )

        data_dict["states"].append(item["pose"])
        data_dict["actions"].append(item["pose"])  # TODO
    data_dict["episode_ends"].append(len(data_dict["imgs"]))


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

    for index, json_file in tqdm(enumerate(json_files)):
        read_json(json_file, data, image_params)

    # colored_clouds = np.array(colored_clouds).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    states = np.array(states).astype(np.float32)

    # for 2d
    imgs = np.array(imgs).astype(np.uint8)

    episode_ends = np.array(episode_ends).astype(np.int64)

    print(actions.shape, states.shape, episode_ends)
    # exit()
    with zarr.open(save_path, mode="w") as zf:
        data_group = zf.create_group("data")
        data_group.create_dataset("action", data=actions, dtype="float32")
        # data_group.create_dataset("point_cloud", data=colored_clouds, dtype="float32")
        data_group.create_dataset("state", data=states, dtype="float32")

        # for 2d
        data_group.create_dataset("img", data=imgs, dtype="uint8")
        data_group.create_dataset("wrist_img", data=wrist_imgs, dtype="uint8")
        data_group.create_dataset("scene_img", data=scene_imgs, dtype="uint8")

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
    params = {
        "rgb": {
            "size": (960, 540),
            "crop": (230, 0, 770, 540),
            "resize": (240, 240),
        },
        "wrist": {
            "size": (640, 480),
            "crop": (0, 0, 640, 480),
            "resize": (240, 240),
        },
        "scene": {
            "size": (640, 480),
            "crop": (100, 160, 540, 480),
            "resize": (240, 240),
        },
    }

    data_path_list = [
        # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1226_random",
        # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1224",
        "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1226_bowl",
    ]
    save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/240_1226_bowl"
    run(data_path_list, save_path, params)

    print("done")


if __name__ == "__main__":
    run_240()
