# WebDreamer: Model-Based Planning for Web Agents

![image](https://github.com/user-attachments/assets/a1189fee-ff43-45fc-a818-3dc6befb6ad2)

## About
This repo contains the code for our paper [*Is Your LLM Secretly a World Model of the Internet*? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559).

Our paper tackles the critical question: “*How to scale inference-time compute for language agents?*” The solution lies in using LLMs as a world model of the internet to predict the outcomes of actions on websites. Our method, **WebDreamer**, employs LLM-based simulation for speculative planning on the web, surpassing reactive baselines while offering greater safety and flexibility compared to tree search methods.

## Results
### Strong performance on VisualWebArena and Mind2Web-live
| Benchmark        | Observation \( O \) | Method                                 | Completion Rate | Success Rate       |
|------------------|----------------------|----------------------------------------|-----------------|--------------------|
| **VisualWebArena** | Screenshot+SoM      | Gemini-1.5-Pro + Reactive (Koh et al., 2024a) | -               | 12.0%             |
|                  |                      | GPT-4 + Reactive (Koh et al., 2024a)   | -               | 16.4%             |
|                  |                      | GPT-4o + Reactive (Koh et al., 2024a)  | -               | 17.7% †           |
|                  |                      | GPT-4o + Tree Search (Koh et al., 2024b) | -             | 26.4%             |
|                  |                      | **GPT-4o + WebDreamer**                   | -               | 23.6% (↑33.3%)    |
| **Mind2Web-live**| HTML                | GPT-4 + Reactive (Pan et al., 2024b)  | 48.8%           | 23.1%             |
|                  |                      | Claude-3-Sonnet + Reactive (Pan et al., 2024b) | 47.9%      | 22.1%             |
|                  |                      | Gemini-1.5-Pro + Reactive (Pan et al., 2024b) | 44.6%      | 22.3%             |
|                  |                      | GPT-4-turbo + Reactive (Pan et al., 2024b) | 44.3%         | 21.1%             |
|                  |                      | GPT-3.5-turbo + Reactive (Pan et al., 2024b) | 40.2%         | 16.5%             |
|                  |                      | GPT-4o + Reactive (Pan et al., 2024b)  | 47.6%           | 22.1%             |
|                  |                      | **GPT-4o + WebDreamer**                   | 49.9%           | 25.0% (↑13.1%)    |

Compared to the reactive baselines, WebDreamer significantly improves performance by 33.3% and 13.1% on VisualWebArena and Mind2Web-live, respectively.

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


## Citation
```
@article{DBLP:journals/corr/abs-2411-06559,
  author    = {Yu Gu and Boyuan Zheng and Boyu Gou and Kai Zhang and Cheng Chang and Sanjari Srivastava and Yanan Xie and Peng Qi and Huan Sun and Yu Su},
  title     = {Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents},
  journal   = {CoRR},
  volume    = {abs/2411.06559},
  year      = {2024},
  url       = {https://arxiv.org/abs/2411.06559},
  eprinttype= {arXiv},
  eprint    = {2411.06559},
}
```
