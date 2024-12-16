import json, os, zarr
import numpy as np


def read_json(json_file, colored_clouds, states, actions, episode_ends):
    with open(json_file, "r") as file:
        data = json.load(file)

    data_path = os.path.dirname(json_file)

    for item in data:
        file = os.path.join(
            data_path, item["point_cloud"].split(data_path.split("/")[-1] + "/")[1]
        )
        colored_cloud = np.load(file)
        colored_clouds.append(colored_cloud)

        states.append(item["pose"])
        actions.append(item["pose"])  # TODO
    episode_ends.append(len(colored_clouds))


def process(save_path, json_files):
    colored_clouds = []
    actions = []
    states = []
    episode_ends = []

    for index, json_file in enumerate(json_files):
        read_json(json_file, colored_clouds, states, actions, episode_ends)

    colored_clouds = np.array(colored_clouds).astype(np.uint8)
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

    # data_path_list = [
    #     # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1211",
    #     "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213"
    # ]
    # save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1213"
    # run(data_path_list, save_path)

    data_path_list = [
        "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213/cube_a29",
        "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213/cube_a17",
    ]
    save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1213_a29_a17"
    run_folder_list(data_path_list, save_path)

    print("done")
