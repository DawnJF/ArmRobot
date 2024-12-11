import json, os, zarr
import numpy as np

def run(data_path, save_path):

    colored_clouds = []
    actions = []
    states = []
    episode_ends = []

    folders_name = os.listdir(data_path)
    json_files = [
        os.path.join(data_path, folder_name, "data.json")
        for folder_name in folders_name
    ]

    for index, json_file in enumerate(json_files):
        with open(json_file, "r") as file:
            data = json.load(file)

        for item in data:
            colored_cloud = np.load()
            colored_clouds.append(colored_cloud)

            states.append(item["pose"])
            actions.append(item["pose"]) # TODO

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

    data_path = "/storage/liujinxin/code/tram/iDP3/data/raw_data/1205_crop/"
    save_path = "/storage/liujinxin/code/tram/iDP3/data/train_data/corn1205"

    run(data_path, save_path)