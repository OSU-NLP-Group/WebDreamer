# WebDreamer: Model-Based Planning for Web Agents
Code for our paper [*Is Your LLM Secretly a World Model of the Internet*? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559v1). :construction: :construction: Under Construction

![image](https://github.com/user-attachments/assets/a1189fee-ff43-45fc-a818-3dc6befb6ad2)


## Results
### Strong performance on VisualWebArena and Mind2Web-live
![image](https://github.com/user-attachments/assets/e2ca2216-848c-4fed-b623-5297795407e0)

### Better efficiency than tree search with true interactions
![image](https://github.com/user-attachments/assets/d736157e-9b09-43e1-b54e-a17666f4382b)


## Structure of this repo
[main](https://github.com/OSU-NLP-Group/WebDreamer): Different modules of WebDreamer that can be played with independently.

[vwa](https://github.com/OSU-NLP-Group/WebDreamer/tree/vwa): Code to reproduce our experiments on VisualWebArena.

[mind2web-live](https://github.com/OSU-NLP-Group/WebDreamer/tree/mind2web-live): Code to reproduce our experiments on Mind2Web-live

## WebDreamer Modules Usage

### World Model
The world model module predicts webpage changes in multiple format (change description, a11y tree, html). 

#### Example Code
```bash
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

```bash
python simulation_scoring.py
```

### Controller

```bash
python controller.py
```


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
