import os
from PIL import Image
import numpy as np

left, top, right, bottom = 230, 0, 770, 540

o_size = (960, 540)


def process_image(img: Image, left, top, right, bottom, new_size=(224, 224)):
    """
    裁剪单张图片并保存为新文件

    参数:
    - left: int, 裁剪区域的左边界
    - top: int, 裁剪区域的上边界
    - right: int, 裁剪区域的右边界
    - bottom: int, 裁剪区域的下边界
    """

    # 裁剪图片
    size = img.size
    img = img.crop((left, top, right, bottom))
    print(f"裁剪完成: {size} -> {img.size}")

    # 设置新的尺寸（例如，宽度=800，高度=600）
    if new_size:
        img = img.resize(new_size, Image.BICUBIC)
        print(f"resize: -> {img.size}")
    return img


def process_image_file(input_file_path, output_file_path, left, top, right, bottom):
    try:
        img = Image.open(input_file_path)
        img = process_image(img, left, top, right, bottom)
        img.save(output_file_path)
    except Exception as e:
        print(f"处理文件 {input_file_path} 时发生错误: {e}")


def process_image_npy(npy):
    try:
        img = Image.fromarray(npy)
        assert o_size == img.size
        img = process_image(img, left, top, right, bottom)
        return np.array(img)
    except Exception as e:
        print(f"发生错误: {e}")


def process_image_file2(npy):
    try:
        img = Image.open(npy)
        assert o_size == img.size
        img = process_image(img, left, top, right, bottom)
        return np.array(img)
    except Exception as e:
        print(f"发生错误: {e}")


def process_folder(input_folder, left, top, right, bottom):
    """
    遍历文件夹中的图片，调用裁剪函数并保存到新文件夹

    参数:
    - input_folder: str, 输入图片文件夹路径
    - output_folder: str, 输出图片文件夹路径
    - left: int, 裁剪区域的左边界
    - top: int, 裁剪区域的上边界
    - right: int, 裁剪区域的右边界
    - bottom: int, 裁剪区域的下边界
    """
    # 确保输出文件夹存在
    # os.makedirs(output_folder, exist_ok=True)

    for dp, _, df in os.walk(input_folder):
        for file in df:
            input_file_path = os.path.join(dp, file)
            output_file_path = input_file_path.replace("/1205/", "/1205_crop/")

            if os.path.isfile(input_file_path) and (
                file.lower().endswith((".json"))
                or input_file_path.find("/scene/") != -1
            ):
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                os.system(f"cp {input_file_path} {output_file_path}")
            elif os.path.isfile(input_file_path) and file.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            ):
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                process_image_file(
                    input_file_path, output_file_path, left, top, right, bottom
                )
            else:
                print(f"跳过非图片文件: {file}")


def run():
    # 示例使用
    input_folder = (
        "/storage/liujinxin/code/tram/iDP3/data/raw_data/1205"  # 输入图片文件夹路径
    )
    # output_folder = "/Users/majianfei/Downloads/1203/cube1/image/rgb_"  # 输出图片文件夹路径
    # output_folder = input_folder.replace("/1205/","/1205_crop/")
    left, top, right, bottom = 250, 150, 720, 404

    process_folder(input_folder, left, top, right, bottom)


def test():
    file = "/storage/liujinxin/code/ArmRobot/dataset/rgb_960x540_1.png"
    output_file = "/storage/liujinxin/code/ArmRobot/dataset/rgb_960x540_1_.png"
    left, top, right, bottom = 230, 0, 770, 540

    process_image_file(file, output_file, left, top, right, bottom)


if __name__ == "__main__":
    # run()
    test()
