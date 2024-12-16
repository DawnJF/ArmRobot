import json, os, zarr
import numpy as np


def run(data_path_list, save_path):

    colored_clouds = []
    actions = []
    states = []
    episode_ends = []

    for data_path in data_path_list:
        folders_name = os.listdir(data_path)
        json_files = [
            os.path.join(data_path, folder_name, "data.json")
            for folder_name in folders_name
        ]

        folder = data_path.split("/")[-1]

        for index, json_file in enumerate(json_files):
            with open(json_file, "r") as file:
                data = json.load(file)

            for item in data:
                file = os.path.join(data_path, item["point_cloud"].split(folder + "/")[1])
                colored_cloud = np.load(file)
                colored_clouds.append(colored_cloud)

                states.append(item["pose"])
                actions.append(item["pose"])  # TODO

            episode_ends.append(len(colored_clouds))

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


if __name__ == "__main__":

    data_path = [
        # "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1211",
    "/storage/liujinxin/code/ArmRobot/dataset/raw_data/1213"]
    save_path = "/storage/liujinxin/code/ArmRobot/dataset/train_data/1213"

    run(data_path, save_path)
    print("done")
