import io
from openai import OpenAI
import os
import base64
import time
from PIL import Image


# base 64 编码格式
def encode_image(image):
    if os.path.isfile(image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    else:
        pil_image = Image.fromarray(image)
        byte_stream = io.BytesIO()
        pil_image.save(byte_stream, format="PNG")

        # 获取 PNG 图像数据
        png_data = byte_stream.getvalue()
        return base64.b64encode(png_data).decode("utf-8")


prompt = """
# 角色：服务型交互机器人

# 个人简介：
- 语言：中文
- 描述：我是一个服务型交互机器人，你能根据用户的问题，进行回答，并根据用户的需求调用相应技能。

## 目标：
- 对用户的问题进行回答
- 对用户提出的需求可以调用相应技能
- 输出回答以及调用的技能

## 约束：
- 提供准确的回答和技能调用。
- 在回答时，确保不改变用户的意图和需求。

## 技能：
- fun_1: 拿起白色方块到碗里。
- fun_2: 把碗递给用户。


## 工作流程：
1. 输入: 接收用户输入的问题。
2. 理解用户的意图：根据用户意图进行对话或技能调用
3. 输出格式: 
    - 对话:
        - 如果用户的意图是对话，根据提供的信息进行回复，要求回复内容流畅且符合已知信息
        - 格式: 正常文本
    - 技能调用:
        - 如果用户提出需求，根据需求调用技能
        - 格式: fun_n+肯定的答复，表示自己正在做这件事情。
"""


class VLM:
    def __init__(self):
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key="sk-524f61a72bed4fc9813fcc4e9b768bb3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.messages = [{"role": "system", "content": prompt}]

    def chat(self, user_input, current_obs=None):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_input},
            ],
        }
        if current_obs:
            base64_image = encode_image(current_obs)
            msg["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

        self.messages.append(msg)
        completion = self.client.chat.completions.create(
            model="qwen-vl-max-latest",
            messages=self.messages,
            response_format={"type": "text"},
        )
        assistant_message = completion.choices[0].message
        self.messages.append(assistant_message.model_dump())
        return assistant_message.content


def test():
    vlm = VLM()

    user_input = input("第一轮对话：")

    response = vlm.chat(user_input, "/Users/majianfei/Downloads/rgb_960x540_1.png")
    print(f"机器人输出：{response}")

    user_input = input("第二轮对话：")
    response = vlm.chat(user_input, "/Users/majianfei/Downloads/rgb_960x540_1.png")
    print(f"机器人输出：{response}")

    user_input = input("第三轮对话（方块已经在碗里了）：")
    response = vlm.chat(user_input, "/Users/majianfei/Downloads/rgb_960x540_0.png")
    print(f"机器人输出：{response}")


if __name__ == "__main__":
    test()
