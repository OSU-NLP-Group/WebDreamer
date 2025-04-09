# WebDreamer: Model-Based Planning for Web Agents

![image](https://github.com/user-attachments/assets/a1189fee-ff43-45fc-a818-3dc6befb6ad2)

## Release Plan
- [x] Release world model [training data](https://huggingface.co/datasets/osunlp/Dreamer-V1-Data)
- [x] Release [checkpoints](https://huggingface.co/osunlp/Dreamer-7B)

## About
This repo contains the code for our paper [*Is Your LLM Secretly a World Model of the Internet*? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559).

Our paper tackles the critical question: “*How to scale inference-time compute for language agents?*” The solution lies in using LLMs as a world model of the internet to predict the outcomes of actions on websites. Our method, **WebDreamer**, employs LLM-based simulation for speculative planning on the web, surpassing reactive baselines while offering greater safety and flexibility compared to tree search methods.
All resources (including training data and resulting models) are available at [HF Collection](https://huggingface.co/collections/osunlp/webdreamer-67ee17325839c8a02339dbfb).

## Results
### Strong performance on VisualWebArena and Mind2Web-live
| Benchmark        | Method                                 | Success Rate       |
|------------------|-----------------|--------------------|
| **VisualWebArena** | GPT-4o + Reactive | 17.6%       |
|  | GPT-4o + Tree Search | 26.2%    |
|  | **GPT-4o + WebDreamer** | 23.6% (↑34.1%) |
| **Online-Mind2Web** | GPT-4o + Reactive | 26.0%       |
|  | **GPT-4o + WebDreamer** | 37.0% (↑42.3%) |
| **Mind2Web-live**   | GPT-4o + Reactive | 20.2%       |
|  | **GPT-4o + WebDreamer**         | 25.0% (↑23.8%)    |

Compared to the reactive baselines, WebDreamer significantly improves performance by 34.1%, 42.3%, and 23.8% on VisualWebArena, Online-Mind2Web, and Mind2Web-live, respectively.

### Better efficiency than tree search with true interactions
<img width="1502" alt="image" src="https://github.com/user-attachments/assets/0afbc22d-b1eb-4026-a167-e1852cde7677">

WebDreamer effectively explores the search space through simulations, which largely reduces the reliance on real-world interactions while maintaining robust performance.


## Structure of this repo
[`main`](https://github.com/OSU-NLP-Group/WebDreamer): Different modules of WebDreamer that can be played with independently.

[`vwa`](https://github.com/OSU-NLP-Group/WebDreamer/tree/vwa): Code to reproduce our experiments on VisualWebArena. :construction:

[`mind2web-live`](https://github.com/OSU-NLP-Group/WebDreamer/tree/mind2web-live): Code to reproduce our experiments on Mind2Web-live. :construction:

## WebDreamer Modules Usage

### World Model
The world model module predicts webpage changes in multiple format (change description, a11y tree, html). 

#### Example Code
```python
world_model = WebWorldModel(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
screenshot_path = "demo_data/shopping_0.png"
screenshot = encode_image(screenshot_path)
screenshot = "data:image/jpeg;base64," + screenshot
action_description = "type 'red blanket' in the search bar and click search"
task = "Buy the least expensive red blanket (in any size) from 'Blankets & Throws' category."

imagination = world_model.multiple_step_change_prediction(screenshot, screenshot_path, task,
                                                          action_description,
                                                          format='accessibility', k=3)
```

#### Parameters
* screenshot_path: Path to the screenshot of the webpage.
* task: Description of the goal to achieve on the webpage.
* action_description: Initial action to perform.
* format: Desired output format for webpage state changes:
  * 'change' for textual descriptions.
  * 'accessibility' for an accessibility tree structure.
  * 'html' for HTML structure of the predicted page.
* k: Number of imagination steps to simulate.


### Simulation Scoring

#### Example Code
```python
screenshot_path = "demo_data/shopping_0.png"
screenshots = [Image.open(screenshot_path)]
actions = ["None"]
action_description_list = [
    "type 'red blanket' in the search bar",
    "click the element Home & Kitchen",
    "type 'kobe' in the search bar",
    "type 'the ohio state university' in the search bar"
]
task = "Buy the least expensive red blanket (in any size)"
scores, simulations = evaluate_simulation(
    screenshots, 
    actions, 
    task, 
    "https://www.amazon.com", 
    action_description_list, 
    num_of_sim=3, 
    num_workers=50, 
    n=10, 
    steps=2
)
```

#### Parameters
* screenshots: List of PIL.Image screenshots representing webpage states.
* actions: List of actions performed by the agent.
* task: Description of the goal to achieve on the webpage.
* url: The current webpage URL.
* action_description_list: List of action descriptions to evaluate.
* num_of_sim: Number of simulations per action. 
* steps: Number of imagination steps per simulation. 
* num_workers: Number of parallel workers for simulations.

### Controller

#### Example Code
```python
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
# Map selected indices back to action descriptions
selected_actions = [action_description_list[int(i)] for i in selected_actions]
```

#### Parameters
* screenshots: List of PIL.Image screenshots representing webpage states.
* actions: List of previously executed actions.
* task: Description of the goal to achieve on the webpage.
* url: The current webpage URL.
* action_description_list: List of action descriptions to evaluate.

### Using Dreamer-7B
We released Dreamer-7B and its VWA in-domain continue trained variants (https://huggingface.co/osunlp/Dreamer-7B).
Please note that current Dreamer-7B only supports image (w/ or w/o SoM) observation space.

## Citation
```
@article{Gu2024WebDreamer,
  author    = {Yu Gu and Kai Zhang and Yuting Ning and Boyuan Zheng and Boyu Gou and Tianci Xue and Cheng Chang and Sanjari Srivastava and Yanan Xie and Peng Qi and Huan Sun and Yu Su},
  title     = {Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents},
  journal   = {CoRR},
  volume    = {abs/2411.06559},
  year      = {2024},
  url       = {https://arxiv.org/abs/2411.06559},
  eprinttype= {arXiv},
  eprint    = {2411.06559},
}
```
