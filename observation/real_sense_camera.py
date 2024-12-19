#!/usr/bin/env python3
import cv2
import sys
import numpy as np
from collections import deque
import pyrealsense2 as rs
from multiprocessing import Process, Manager
import time
import multiprocessing

sys.path.insert(0, "/home/robot/ArmRobot")
from observation.depth_image_process import process_depth_image

multiprocessing.set_start_method("fork")

np.printoptions(3, suppress=True)


def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [
        devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))
    ]
    devices.sort()  # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices


def init_given_realsense(
    device,
    enable_rgb=True,
    enable_depth=True,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        # Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        h, w = 768, 1024
        print(f"==== enable_stream: w:{w} h:{h} 30 z16 ====")
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        h, w = 540, 960
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)

    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        # Set the inter-camera sync mode
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)

        # set min distance
        depth_sensor.set_option(rs.option.min_distance, 0.05)

        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)

        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        print(f"camera intrinsics: {intrinsics}")
        assert intrinsics.width == 1024 and intrinsics.height == 768
        assert intrinsics.fx == 731.6640625 and intrinsics.fy == 731.4296875
        assert intrinsics.ppx == 510.51171875 and intrinsics.ppy == 418.9140625

        camera_info = CameraInfo(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )

        print("camera {} init.".format(device))
        print("depth_scale: ", depth_scale)
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None


class CameraInfo:
    """Camera intrisics for point cloud creation."""

    def __init__(self, width, height, fx, fy, cx, cy, scale=1):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


class SingleVisionProcess(Process):
    def __init__(
        self,
        device,
        data,
        enable_rgb=True,
        enable_depth=False,
        sync_mode=0,
    ) -> None:
        super(SingleVisionProcess, self).__init__()
        self.data = data
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        self.sync_mode = sync_mode

    def get_vision(self):
        frame = self.pipeline.wait_for_frames()

        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())

            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())

            clip_lower = 0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high

            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)

            point_cloud_frame = process_depth_image(
                self.camera_info, color_frame, depth_frame
            )

        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = None
            point_cloud_frame = None

        # print("color:", color_frame.shape)
        # print("depth:", depth_frame.shape)

        return color_frame, depth_frame, point_cloud_frame

    def run(self):
        self.pipeline, self.align, self.depth_scale, self.camera_info = (
            init_given_realsense(
                self.device,
                enable_rgb=self.enable_rgb,
                enable_depth=self.enable_depth,
                sync_mode=self.sync_mode,
            )
        )

        debug = False
        while True:
            color_frame, depth_frame, point_cloud_frame = self.get_vision()
            self.data[0] = color_frame
            self.data[1] = depth_frame
            self.data[2] = point_cloud_frame
            time.sleep(0.016)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()


class MultiRealSense(object):
    def __init__(
        self,
        front_cam_idx=0,
    ):
        print("==== MultiRealSense ====")

        self.devices = get_realsense_id()

        # 保留最新的数据
        self.manager = Manager()
        self.data = self.manager.list([None, None, None])

        # 0: f1380328, 1: f1422212

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)

        self.front_process = SingleVisionProcess(
            self.devices[front_cam_idx],
            self.data,
            enable_rgb=True,
            enable_depth=True,
            sync_mode=1,
        )

        self.front_process.start()
        print("==== camera start.")

    def __call__(self):
        cam_dict = {}

        front_color = self.data[0]
        front_depth = self.data[1]
        point_cloud_frame = self.data[2]
        cam_dict.update(
            {
                "color": front_color,
                "depth": front_depth,
                "point_cloud": point_cloud_frame,
            }
        )

        return cam_dict

    def finalize(self):

        self.front_process.terminate()

    def __del__(self):
        self.finalize()
        self.manager.shutdown()


def show_depth(depth):
    depth_frame = depth
    # depth_frame = depth.numpy()
    # clip_lower =  0.01
    # clip_high = 1.0
    # depth_frame = depth_frame.astype(np.float32)
    # depth_frame[depth_frame < clip_lower] = clip_lower
    # depth_frame[depth_frame > clip_high] = clip_high

    normalized_depth = depth_frame * 255
    depth_image = normalized_depth.astype(np.uint8)

    # Use OpenCV to display the depth image
    cv2.imshow("Depth Image", depth_image)

    cv2.waitKey(1)

def show_point_cloud(point_cloud):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 假设你的点云数据在 point_cloud 变量中
    # point_cloud 的形状为 (4096, 6)，前 3 列是坐标，后 3 列是颜色（可选）
    # point_cloud = np.random.rand(4096, 6)  # 示例随机数据

    # 提取点云坐标
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]

    # 提取颜色信息（如果有）
    if point_cloud.shape[1] == 6:
        colors = point_cloud[:, 3:6]
    else:
        colors = None

    # 可视化点云
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=1, cmap='viridis')  # s=1 是点的大小

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.colorbar(sc, label='Color') if colors is not None else None
    plt.show()

def show_img(img):

    # Use OpenCV to display the depth image
    cv2.imshow("Depth Image", img)

    cv2.waitKey(1)


if __name__ == "__main__":
    cam = MultiRealSense()
    import matplotlib.pyplot as plt

    while True:
        out = cam()
        print(out.keys())
        if out["color"] is None:
            time.sleep(0.2)
            continue
        print("color: ", out["color"].shape)
        print("depth: ", out["depth"].shape)
        print("point_cloud: ", out["point_cloud"].shape)

        # imageio.imwrite(f'/media/robot/2CCF4D6BBC2D923E/mpz/color.png', out['color'])
        # imageio.imwrite(f'color_right.png', out['right_color'])
        # imageio.imwrite(f'/media/robot/2CCF4D6BBC2D923E/mpz/depth.png', out['depth'])
        # imageio.imwrite(f'depth_front.png', out['right_front'])
        # cv2.imwrite(f'/media/robot/2CCF4D6BBC2D923E/mpz/color.png', out['color'])
        # cv2.imwrite(f'/media/robot/2CCF4D6BBC2D923E/mpz/depth.png', out['depth'])
        show_depth(out["depth"])
        # show_point_cloud(out["point_cloud"])
        # show_img(out["color"])
        # plt.savefig("front_depth.png")
        # import visualizer
        # visualizer.visualize_pointcloud(out['right_point_cloud'])
        # visualizer.visualize_pointcloud(out['point_cloud'])
        time.sleep(0.01)
