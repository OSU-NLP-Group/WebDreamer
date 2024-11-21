from typing import Optional
import base64
from io import BytesIO
import os
from openai import OpenAI
import numpy as np
from PIL import Image
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from world_model import WebWorldModel

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def pil_to_b64(img: Image.Image) -> str:
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def evaluate_success_with_action(
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        action_description: str,
        intent: str,
        models: list[str],
        intent_images: Optional[Image.Image] = None,
        n: int = 20,
        top_p: float = 1.0
) -> float:
    last_actions_str = '\n'.join(actions)
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
Propopsed Action: {action_description}
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
Proposed Action: {action_description}
"""
        })

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, your goal is to decide **whether the proposed action successfully accomplish the task**. If it does not but is on the right track towards success, you should also output as such. You don't have the ability to predict the future, so you should only consider the current state of the webpage and the proposed action.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]

    all_responses = []
    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=256,
            top_p=top_p,
            n=n // len(models)
        )
        all_responses.extend(response.choices)

    all_scores = []
    message_content = None
    for r_idx, r in enumerate(all_responses):
        # print(r.message.content)
        if message_content is None:
            message_content = r.message.content
        try:
            pred = re.search(r'Status: "?(.+)"?', r.message.content).group(1)
            if 'success' in pred.lower():
                score = 1.0
            else:
                # Check if it's on the path to success
                on_path = re.search(r'On the right track to success: "?(.+)"?', r.message.content).group(1)
                if 'yes' in on_path.lower():
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0

        all_scores.append(score)

    score = np.mean(all_scores)
    return score, message_content

def evaluate_simulation_inner(
        screenshots: list[Image.Image],
        actions: list[str],
        current_url: str,
        imagination: str,
        intent: str,
        models: list[str],
        intent_images: Optional[Image.Image] = None,
        n: int = 20,
        top_p: float = 1.0
) -> float:
    last_actions_str = '\n'.join(actions)
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
Simulated steps: {imagination}
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
Simulated steps: {imagination}
"""
        })

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the current state of the webpage, your goal is to decide **whether the simulated steps by the agent indicate a successful execution of the user intent**. In particular, if the predicted state (i.e., the current state represented by the last image plus all the predicted changes so far) corresponds to a successful final state. If it is a failure but it looks like the simulated steps are on the right track towards success, you should also output as such. Note that, in the simulated steps, all the state changes are predicted by the agent's world model, and they may not actually be faithful to the real website interactions (e.g., some proposed actions may not be avaiable in a realistic website). You should also account for this in your evaluation (e.g., if the predicted state changes are not reasonable then it's probably a failure).

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]

    all_responses = []
    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=256,
            top_p=top_p,
            n=n // len(models)
        )
        all_responses.extend(response.choices)

    all_scores = []
    for r_idx, r in enumerate(all_responses):
        # print(r.message.content)
        try:
            pred = re.search(r'Status: "?(.+)"?', r.message.content).group(1)
            if 'success' in pred.lower():
                score = 1.0
            else:
                # Check if it's on the path to success
                on_path = re.search(r'On the right track to success: "?(.+)"?', r.message.content).group(1)
                if 'yes' in on_path.lower():
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0

        all_scores.append(score)

    score = np.mean(all_scores)
    return score


def evaluate_success(screenshots: list[Image.Image], actions: list[str], current_url: str, last_reasoning: str,
                     intent: str, models: list[str], intent_images: Optional[Image.Image] = None,
                     n: int = 20, top_p: float = 1.0, should_log: bool = False) -> float:
    """Compute the value of a state using the value function.

    Args:
        state (str): The state to compute the value of.
        action (list[str]): The action to take in the state.
        intent (str): The intent to compute the value of.
        intent_images (list[Image.Image], optional): The images corresponding to the intent. Defaults to None.
        file_prefix (str, optional): The prefix to use for the file name. Defaults to ''.
    Returns:
        float: The value of the state.
    """
    last_actions_str = '\n'.join(actions[:-1])
    last_response = actions[-1]
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
Bot response to the user: {last_response}
Current URL: {current_url}
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
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
Bot response to the user: {last_response}
Current URL: {current_url}
The images corresponding to the user intent are shown in the FIRST {len(intent_images)} images (before the User Intent).
The last {len(screenshots)} snapshots of the agent's trajectory are shown in the LAST {len(screenshots)} images. The LAST IMAGE represents the current state of the webpage.
"""
        })

    messages = [
        {
            "role": "system",
            "content": f"""
You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's intent, the agent's action history, the final state of the webpage, and the agent's response to the user, your goal is to decide whether the agent's execution is successful or not. If the current state is a failure but it looks like the agent is on the right track towards success, you should also output as such.

There are three types of tasks:
1. Information seeking: The user wants to obtain certain information from the webpage, such as the information of a product, reviews, the text in a comment or post, the date of a submission, etc. This may be formulated in the intent as "tell me", "what is", or "list out". The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and respond with the error content, the task is considered to be a failure. It is VERY IMPORTANT that the bot response is the stop action with the correct output. If the bot response is not stop (e.g., it is click, type, or goto), it is considered a failure for information seeking tasks.
2. Site navigation: The user wants to navigate to a specific page (which may also be specified in the intent as "find", "show me", "navigate to"). Carefully examine the agent's action history and the final state of the webpage (shown in the LAST IMAGE) to determine whether the agent successfully completes the task. It is VERY IMPORTANT that the agent actually navigates to the specified page (reflected by the final state of the webpage, in the LAST IMAGE) and NOT just output the name of the item or post. Make sure that the final url is compatible with the task. For example, if you are tasked to navigate to a comment or an item, the final page and url should be that of the specific comment/item and not the overall post or search page. If asked to navigate to a page with a similar image, make sure that an image on the page is semantically SIMILAR to the intent image. If asked to look for a particular post or item, make sure that the image on the page is EXACTLY the intent image. For this type of task to be considered successful, the LAST IMAGE and current URL should reflect the correct content. No need to consider the agent's response.
3. Content modification: The user wants to modify the content of a webpage or configuration. Ensure that the agent actually commits to the modification. For example, if the agent writes a review or a comment but does not click post, the task is considered to be a failure. Carefully examine the agent's action history and the final state of the webpage to determine whether the agent successfully completes the task. No need to consider the agent's response.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
On the right track to success: "yes" or "no"
"""
        },
        {
            "role": "user",
            "content": content
        }
    ]

    all_responses = []
    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=256,
            top_p=top_p,
            n=n // len(models)
        )
        all_responses.extend(response.choices)

    if should_log:
        print('=' * 30)
        print("Value function input:", content[-1])
    all_scores = []
    for r_idx, r in enumerate(all_responses):
        if should_log:
            print(f"Output {r_idx}: {r.message.content}")
        try:
            pred = re.search(r'Status: "?(.+)"?', r.message.content).group(1)
            if 'success' in pred.lower():
                score = 1.0
            else:
                # Check if it's on the path to success
                on_path = re.search(r'On the right track to success: "?(.+)"?', r.message.content).group(1)
                if 'yes' in on_path.lower():
                    score = 0.5
                else:
                    score = 0.0
        except Exception as e:
            print(f"Error parsing response: {e}")
            score = 0.0

        all_scores.append(score)

    score = np.mean(all_scores)
    if should_log:
        print(f"Final score: {score}")
        print('=' * 30)
    return score


def single_action_simulation(screenshots,
                             converted_screenshot,
                             converted_screenshot_path,  # this is only for the sft model
                             actions, task, url, action_description, sim_idx,
                             models=["gpt-4o"], steps=1,
                             n=10):
    simulation_results = {}
    world_model = WebWorldModel(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    score, count = 0, 0
    start_time = time.time()
    try:
        imagination = world_model.multiple_step_change_prediction(converted_screenshot, converted_screenshot_path,
                                                                  task, action_description, k=steps)
        score += evaluate_simulation_inner(screenshots, actions, url, imagination, task, models, n=n)
    except:
        return None
    end_time = time.time()
    duration = (end_time - start_time) % 60
    return {
        'action_description': action_description,
        'imagination': imagination,
        'score': score,
        'duration': duration,
        'sim_idx': sim_idx
    }


def evaluate_simulation(screenshots, actions, task, url, action_description_list,
                        models=["gpt-4o"],
                        num_of_sim=3,
                        steps=1,
                        n=10, num_workers=15):
    # this is for the sft model
    screenshots[-1].save("last_screenshot.png", "PNG")
    converted_screenshot = pil_to_b64(screenshots[-1])

    preds = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for action_description in action_description_list:
            for sim_idx in range(num_of_sim):
                futures.append(executor.submit(single_action_simulation, screenshots,
                                               converted_screenshot,
                                               "last_screenshot.png",  # this is for the sft model
                                               actions, task, url, action_description, sim_idx, models, steps, n))

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                preds.append(result)

    # Organize the output
    all_scores = {}
    all_simulations = {}
    for item in preds:
        action_description = item['action_description']
        if action_description not in all_scores:
            all_scores[action_description] = [item['score']]
        else:
            all_scores[action_description].append(item['score'])

        if action_description not in all_simulations:
            all_simulations[
                action_description] = f"Simulation for action {action_description}:\n{item['imagination']}\nScore: {item['score']}\n\n"
        else:
            all_simulations[
                action_description] += f"Simulation for action {action_description}:\n{item['imagination']}\nScore: {item['score']}\n\n"

    avg_scores = {}
    for action_description in all_scores:
        action_scores = all_scores[action_description]
        if len(action_scores) > 0:
            avg_scores[action_description] = sum(action_scores) / len(action_scores)
        else:
            avg_scores[action_description] = 0

    return avg_scores, all_simulations


if __name__ == "__main__":
    # Test the value function
    screenshot_path = "demo_data/shopping_0.png"
    screenshots = [Image.open(screenshot_path)]
    actions = ["None"]
    action_description = "type 'red blanket' in the search bar"
    task = "Buy the least expensive red blanket (in any size)"
    action_description_list = [
        "type 'red blanket' in the search bar",
        "click the element Home & Kitchen",
        "type 'kobe' in the search bar",
        "type 'the ohio state university' in the search bar"
    ]
    scores = evaluate_simulation(screenshots, actions, task, "https://www.amazon.com", action_description_list,
                                 num_of_sim=3, num_workers=50, n=10, steps=2)
    print(scores)
