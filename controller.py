from typing import Tuple, Optional
import base64
from io import BytesIO
import os
import re
from openai import OpenAI
import numpy as np
from PIL import Image
import requests
import re
import time
import random

from copy import deepcopy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from world_model import WebWorldModel

try:
    from vertexai.preview.generative_models import Image as VertexImage
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image')

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def pil_to_vertex(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_vertex = VertexImage.from_bytes(byte_data)
    return img_vertex


def select_actions(screenshots, actions, intent, current_url, action_description_list, intent_images=None,
                   model="gpt-4o"):
    last_actions_str = '\n'.join(actions)
    action_description_list = deepcopy(action_description_list)
    for i, action in enumerate(action_description_list):
        action_description_list[i] = f"{i}: {action}"
    action_descriptions = ';'.join(action_description_list)
    if intent_images is None:
        content = []
        for screenshot in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "detail": "high"
                },
            })

        content.append({
            "type": "text",
            "text": f"""User Intent: {intent}
Action History: {last_actions_str}
Current URL: {current_url}
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
Candidate actions: {action_descriptions}
"""
        })

    else:
        content = []
        for img in intent_images:
            content.extend([
                {
                    "type": "image_url",
                    "image_url": {
                        "url": pil_to_b64(img)
                    },
                }
            ])
        content.append({
            "type": "text",
            "text": f"\nUser Intent: {intent}\n"
        })

        for screenshot in screenshots:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": pil_to_b64(screenshot),
                    "detail": "high"
                },
            })

        content.append({
            "type": "text",
            "text": f"""
Action History: {last_actions_str}
Current URL: {current_url}
The images corresponding to the user intent are shown in the FIRST {len(intent_images)} images (before the User Intent).
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the LAST {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
Proposed Action: {action_descriptions}
"""
        })

    messages = [
        {
            "role": "system",
            "content": f"""
You are assiting a web navigation agent to help a human user navigate a website to complete a task. Given the user's intent, the action history, and the current state of the webpage, the agent has proposed a set of candidate actions to take at the current step. 
Your role is not to determine a best action for the agent at this step, but to filter out the actions that are very likely not relevant or helpful for the agent to accomplish the task.
Please select all actions that you think that could possibly lead the agent to accomplish the task. It's important to note that to accomplish a task, the agent will execute a sequence of actions. So the action to take at this step does not have to immediately lead to the completion of the task. You should select any action that could be relevant for the agent to take in the current state of the webpage. Try to be as thoughtful and comprehensive as you can! Don't miss any possible action. If there is one action that is clearly the best, and all other actions are clearly not very relevant, you can only select one action. Please do this sparely, since some actions may be helpful in a longer horizon. 
A action should be included as long as it could be relevant to the task, even if it may not be the most direct action to take at this step!! Some relevant actions might seem indirect at the first glance, but could be helpful in a longer horizon. Please also include those actions.
Please at least select one action.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process. You must explicitly evaluate each action one by one and imagine whether it could be relevant to the task following the format: actoin:... rationale:...>
Selected actions: id0;id1;aid2; ... 
(please return the index of the action in the candidate actions list, starting from 0. Don't output the action description itself. Separate the indices with semicolons. Do not add spaces or any other characters between after the semicolons.)
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512
    )

    message_content = None
    selected_actions = []
    if message_content is None:
        message_content = response.choices[0].message.content
    print("message_content:", message_content)
    try:
        # use regex to extract the selected actions
        selected_actions = re.findall(r"Selected actions: (.+)", message_content)[0].split(";")
    except Exception as e:
        print(f"Error parsing response: {e}")
        score = 0.0

    return selected_actions


if __name__ == "__main__":
    screenshot_path = "demo_data/shopping_0.png"
    screenshots = [Image.open(screenshot_path)]
    actions = ["None"]  # previous actions so far

    action_description = "type 'red skirt' in the search bar"
    task = "Buy the least expensive red skirt (in any size) on Amazon."

    action_description_list = [
        "type 'red skirt' in the search bar",
        "click the element Women Clothes",
        "type 'kobe' in the search bar",
        "type 'the ohio state university' in the search bar"
    ]

    random.shuffle(action_description_list)
    selected_actions = select_actions(screenshots, actions, task, "https://www.amazon.com", action_description_list)
    print(selected_actions)
    # get action descriptions from action_str
    selected_actions = [action_description_list[int(i)] for i in selected_actions]
    print(selected_actions)
