

## Installation

Install conda env and packages for both learning and deployment machines:

    conda remove -n idp3 --all
    conda create -n idp3 python=3.8
    conda activate idp3
    
    # for cuda >= 12.1
    pip3 install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121
    # else, 
    # just install the torch version that matches your cuda version
    
    

    # install my visualizer
    cd third_party
    cd visualizer && pip install -e . && cd ..
    pip install kaleido plotly open3d tyro termcolor h5py
    cd ..


    # install 3d diffusion policy
    pip install --no-cache-dir wandb ipdb gpustat visdom notebook mediapy torch_geometric natsort scikit-video easydict pandas moviepy imageio imageio-ffmpeg termcolor av open3d dm_control dill==0.3.5.1 hydra-core==1.2.0 einops==0.4.1 diffusers==0.11.1 zarr==2.12.0 numba==0.56.4 pygame==2.1.2 shapely==1.8.4 tensorboard==2.10.1 tensorboardx==2.5.1 absl-py==0.13.0 pyparsing==2.4.7 jupyterlab==3.0.14 scikit-image yapf==0.31.0 opencv-python==4.5.3.56 psutil av matplotlib setuptools==59.5.0

    cd Improved-3D-Diffusion-Policy
    pip install -e .
    cd ..

    # install for diffusion policy if you want to use image-based policy
    pip install timm==0.9.7

    # install for r3m if you want to use image-based policy
    cd third_party/r3m
    pip install -e .
    cd ../..


[Install on Deployment Machine] Install realsense package for deploy:

    # first, install realsense driver
    # check this version for RealSenseL515: https://github.com/IntelRealSense/librealsense/releases/tag/v2.54.2

    # also install python api
    pip install pyrealsense2==2.54.2.5684

## Usage

**Train.** The script to train policy:

    # 3d policy
    bash scripts/train_policy.sh idp3 gr1_dex-3d 0913_example

**Deploy.** After you have trained the policy, deploy the policy with the following command. For missing packages such as `communication.py`, see another [our repo](https://github.com/YanjieZe/Humanoid-Teleoperation/tree/main/humanoid_teleoperation/teleop-zenoh)

    # 3d policy
    bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example


Note that you may not run the deployment code without a robot (differet robots have different API). The code we provide is more like an example to show how to deploy the policy. You could modify the code to fit your own robot (any robot with a camera is OK).

**Visualize.** You can visualize our training data example by running (remember to set the dataset path):

    bash scripts/vis_dataset.sh

You can specify `vis_cloud=1` to render the point cloud as in the paper.
