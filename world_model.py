import re
import os
import anthropic
from openai import OpenAI
import base64
from llms.utils import openai_messages_to_claude


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def parse_proposed_action(text):
    operation_type_pattern = r'OPERATION TYPE:\s*([\s\S]+?)\n\n|OPERATION TYPE:\s*([\s\S]+)'
    element_pattern = r'ELEMENT:\s*([\s\S]+?)\n\n|ELEMENT:\s*([\s\S]+)'

    match = re.search(operation_type_pattern, text)
    if match:
        operation_type = match.group(1) if match.group(1) else match.group(2)
    else:
        operation_type = None

    match = re.search(element_pattern, text)
    if match:
        element = match.group(1) if match.group(1) else match.group(2)
    else:
        element = None

    return operation_type, element


class WebWorldModel:
    def __init__(self, client):
        self.client = client
    def state_change_prediction_in_website(self, screenshot, task, action_description, format="change"):
        prompt_text = " Please predict the changes after {}.".format(action_description)
        if format == "change":
            content = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the changes that will occur on the webpage after the operation is performed, such as the appearance of new elements, the disappearance of existing elements, or changes in the content of existing elements. The operation type and the element to operate will be provided in the prompt. Directly output 'State changes: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        elif format == "html":
            content = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the state of the webpage after the operation is performed. In particular, you should generate the html code for the new webpage, highlight the most likely elements appearing in the new page. The operation type and the element to operate will be provided in the prompt. Directly output 'New webpage: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        elif format == "accessibility":
            content = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage and an operation to perform on the webpage. You are required to predict the state of the webpage after the operation is performed. In particular, you should describe the new webpage as an accessibility tree, highlight the most likely elements appearing in the new page. The operation type and the element to operate will be provided in the prompt. Directly output 'New webpage: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        input_messages = [
            {
                "role": "system",
                "content": content
            },
            {
                "role": "user", "content":
                [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {
                        "url": screenshot,
                        "detail": "high"
                    }},
                ]},

        ]

        if type(self.client) == anthropic.Anthropic:
            # for claude
            input_messages, system_message = openai_messages_to_claude(input_messages)
            resp = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                messages=input_messages,
                max_tokens=2048,
                system=system_message
            )
            output = resp.content[0].text
        else:
            # for openai
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=input_messages
            )
            output = [item.message.content for item in response.choices][0]
        return output

    def action_proposal_in_imagination(self, screenshot, task, imaginations, format="change"):
        if format == "change":
            prompt_text = "The above image is a screenshot of a web page. You are required to complete the following task: {}\n ".format(
                task)
        else:
            prompt_text = "You are required to complete the following task: {}\n ".format(task)
        prompt_text += "PREVIOUS ACTIONS: \n"
        for i, item in enumerate(imaginations):
            # todo: sometimes item[0] (i.e., the proposed action) can be None
            prompt_text += str(i + 1) + " : " + item[0] + "\n"
        prompt_text += "\n"

        if format == "change":
            prompt_text += "The above image is the screenshot before actually performing all the prevoius actions. The current webpage should be a combination of the initial screenshot with the following changes."
            prompt_text += "The webpage has gone through several changes caused by previous actions:\n\n".format(
                task)
            for i, item in enumerate(imaginations):
                prompt_text += "ACTION {}: \n{}\n{}\n\n".format(str(i + 1), item[0], item[1])

            prompt_text += "Based on the initial screenshot and the changes to the webpage, please predict a single next step action to complete the given task. When proposing a new action, you must be faithful to the current state of the webpage (given by the initial screenshot and a series of state changes afterwards). This means the action proposed by you must be available given the current webpage state based on your evaluation. Please don't repeat any action from PREVIOUS ACTIONS. Please directly specify the operation in a short natural language description, including the operation type and the element to operate. Don't output anything else."

            input_messages = [
                {"role": "system",
                 "content": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue. Available operation types include \n```click```: Use this to click.\n```type```: Use this to type the content into a field.\n```hover```: Hover over an element.\n```press```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n```scroll down``` or ```scroll up```: Scroll the page up or down.\n\nURL Navigation Actions:\n```goto```: Navigate to a specific URL.\n```go_back```: Navigate to the previously viewed page.\n```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n```stop```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket."},
                {"role": "user",
                 "content":
                     [
                         {"type": "text", "text": prompt_text},
                         {"type": "image_url", "image_url": {
                             "url": screenshot,
                             "detail": "high"
                         }},
                     ]},
            ]
        else:
            last_step = imaginations[-1]
            prompt_text += "The current state of the webpage is as follows: \n{}\n".format(last_step[1])
            prompt_text += "Based on the current state of the webpage, please predict a single next step action to complete the given task. When proposing a new action, you must be faithful to the current state of the webpage. This means the action proposed by you should be available given the current webpage state based on your evaluation. Please don't repeat any action from PREVIOUS ACTIONS. Please directly specify the operation in a short natural language description, including the operation type and the element to operate. Don't output anything else."
            input_messages = [
                {"role": "system",
                 "content": "You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue. Available operation types include \n```click```: Use this to click.\n```type```: Use this to type the content into a field.\n```hover```: Hover over an element.\n```press```:  Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).\n```scroll down``` or ```scroll up```: Scroll the page up or down.\n\nURL Navigation Actions:\n```goto```: Navigate to a specific URL.\n```go_back```: Navigate to the previously viewed page.\n```go_forward```: Navigate to the next page (if a previous 'go_back' action was performed).\n\nCompletion Action:\n```stop```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket."},
                {"role": "user",
                 "content": prompt_text}
            ]

        if type(self.client) == anthropic.Anthropic:
            # for claude
            input_messages, system_message = openai_messages_to_claude(input_messages)
            resp = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                messages=input_messages,
                max_tokens=2048,
                system=system_message
            )
            output = resp.content[0].text
        else:
            # for openai
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=input_messages
            )
            output = [item.message.content for item in response.choices][0]
        return output

    def state_change_prediction_in_imagination(self, screenshot, task, imaginations, action, format="change"):
        prompt_text = " Please predict the changes after action: {}".format(action)
        for i, item in enumerate(imaginations):
            prompt_text += "ACTION {}: \n{}\n{}\n\n".format(str(i + 1), item[0], item[1])
        prompt_text += "Based on the initial screenshot and the changes to the webpage, please predict the changes after action: {}".format(
            action)

        if format == "change":
            content = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage, a sequence of actions and state changes applied to the initial screenshot, and an operation to perform on the webpage. You are required to predict the new changes that will occur on the webpage after the operation is performed, such as the appearance of new elements, the disappearance of existing elements, or changes in the content of existing elements. The operation type and the element to operate will be provided in the prompt. Directly output 'State changes: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        elif format == "html":
            content = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage, a sequence of actions and state changes applied to the initial screenshot, and an operation to perform on the webpage. You are required to predict the state of the webpage after the operation is performed. In particular, you should generate the html code for the new webpage, highlight the most likely elements appearing in the new page. The operation type and the element to operate will be provided in the prompt. Directly output 'New webpage: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."
        elif format == "accessibility":
            content = "You are an agent that predicts the effect of an action on a webpage. You will be given a screenshot of a webpage, a sequence of actions and state changes applied to the initial screenshot, and an operation to perform on the webpage. You are required to predict the state of the webpage after the operation is performed. In particular, you should describe the new webpage as an accessibility tree, highlight the most likely elements appearing in the new page. The operation type and the element to operate will be provided in the prompt. Directly output 'New webpage: ...' and don't output anything else. Try to be as comprehensive and detailed as possible."

        input_messages = [
            {
                "role": "system",
                "content": content
            },
            {"role": "user",
             "content":
                 [
                     {"type": "text", "text": prompt_text},
                     {"type": "image_url", "image_url": {
                         "url": screenshot,
                         "detail": "high"
                     }},
                 ]},
        ]

        if type(self.client) == anthropic.Anthropic:
            # for claude
            input_messages, system_message = openai_messages_to_claude(input_messages)
            resp = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                messages=input_messages,
                max_tokens=2048,
                system=system_message
            )
            output = resp.content[0].text
        else:
            # for openai
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=input_messages
            )
            output = [item.message.content for item in response.choices][0]
        return output

    def multiple_step_change_prediction(self,
                                        screenshot,
                                        screenshot_path,  # this is for the sft model only
                                        task,
                                        action_description,
                                        format="change",
                                        k=0):
        rtn_str = ''
        imagination_list = []
        rtn_str += "Proposed New Action: \n"
        rtn_str += action_description + '\n'

        # Initial State-Change on webpage
        change_description = self.state_change_prediction_in_website(screenshot, task, action_description, format)
        rtn_str += "Predicted new webpage: \n"
        rtn_str += change_description + '\n'
        imagination_list.append([action_description, change_description])

        # State-Change in imagination
        for i in range(k):
            rtn_str += "==" * 25 + "STEP: " + str(i) + "==" * 25 + '\n'
            # Propose action in the imagination world
            proposed_action_output = self.action_proposal_in_imagination(screenshot, task, imagination_list, format)
            rtn_str += "Proposed New Action: \n"
            rtn_str += proposed_action_output + '\n'
            proposed_action = proposed_action_output
            # State-Change prediction in the imagination world
            imagined_state_change = self.state_change_prediction_in_imagination(screenshot, task, imagination_list,
                                                                                proposed_action, format)
            rtn_str += "Predicted new webpage: \n"
            rtn_str += imagined_state_change + '\n'

            if "stop" in proposed_action_output.lower():
                break
            imagination_list.append([proposed_action, imagined_state_change])

        return rtn_str


if __name__ == '__main__':
    world_model = WebWorldModel(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    screenshot_path = "demo_data/shopping_0.png"
    screenshot = encode_image(screenshot_path)
    screenshot = "data:image/jpeg;base64," + screenshot
    action_description = "type 'red blanket' in the search bar and click search"
    task = "Buy the least expensive red blanket (in any size) from 'Blankets & Throws' category."
    imagination = world_model.multiple_step_change_prediction(screenshot, screenshot_path, task,
                                                              action_description,
                                                              format='accessibility', k=3)

    print(imagination)
