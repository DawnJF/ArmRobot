import json, os, zarr
import numpy as np
from observation.Image_process_utils import process_image_npy


def read_json(json_file, data_dict):
    with open(json_file, "r") as file:
        data = json.load(file)

    data_path = os.path.dirname(json_file)

    for item in data:
        file = os.path.join(
            data_path, item["point_cloud"].split(data_path.split("/")[-1] + "/")[1]
        )
        colored_cloud = np.load(file)
        data_dict["colored_clouds"].append(colored_cloud)

        rgb_img_file = os.path.join(
            data_path, item["rgb"].split(data_path.split("/")[-1] + "/")[1]
        )
        data_dict["imgs"].append(process_image_npy(np.load(rgb_img_file), "rgb"))

        # FIXME
        wrist_img_file = os.path.join(
            data_path,
            rgb_img_file.replace("rgb", "wrist").replace("960x540", "640x480"),
        )
        data_dict["wrist_imgs"].append(
            process_image_npy(np.load(wrist_img_file), "wrist")
        )
        scene_img_file = os.path.join(
            data_path,
            rgb_img_file.replace("rgb", "scene").replace("960x540", "640x480"),
        )
        data_dict["scene_imgs"].append(
            process_image_npy(np.load(scene_img_file), "scene")
        )

        data_dict["states"].append(item["pose"])
        data_dict["actions"].append(item["pose"])  # TODO
    data_dict["episode_ends"].append(len(data_dict["colored_clouds"]))


def process(save_path, json_files):
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

    for index, json_file in enumerate(json_files):
        read_json(json_file, data)

    colored_clouds = np.array(colored_clouds).astype(np.float32)
    actions = np.array(actions).astype(np.float32)
    states = np.array(states).astype(np.float32)

    # for 2d
    imgs = np.array(imgs).astype(np.uint8)

    episode_ends = np.array(episode_ends).astype(np.int64)

    print(colored_clouds.shape, actions.shape, states.shape, episode_ends)
    # exit()
    with zarr.open(save_path, mode="w") as zf:
        data_group = zf.create_group("data")
        data_group.create_dataset("action", data=actions, dtype="float32")
        data_group.create_dataset("point_cloud", data=colored_clouds, dtype="float32")
        data_group.create_dataset("state", data=states, dtype="float32")

        # for 2d
        data_group.create_dataset("img", data=imgs, dtype="uint8")
        data_group.create_dataset("wrist_img", data=wrist_imgs, dtype="uint8")
        data_group.create_dataset("scene_img", data=scene_imgs, dtype="uint8")

        data_group = zf.create_group("meta")
        data_group.create_dataset("episode_ends", data=episode_ends, dtype="int64")


def run(data_path_list, save_path):

    for data_path in data_path_list:
        folders_name = os.listdir(data_path)
        json_files = [
            os.path.join(data_path, folder_name, "data.json")
            for folder_name in folders_name
        ]
        process(save_path, json_files)


def run_folder_list(json_folder_list, save_path):

    json_files = [
        os.path.join(folder_name, "data.json") for folder_name in json_folder_list
    ]
    process(save_path, json_files)


if __name__ == "__main__":

    data_path_list = [
        # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1211",
        "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1219",
    ]
    save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1219"
    run(data_path_list, save_path)

    # data_path_list = [
    #     "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213/cube_a29",
    #     "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213/cube_a17",
    # ]
    # save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1213_a29_a17"
    # run_folder_list(data_path_list, save_path)

    print("done")
