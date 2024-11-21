# WebDreamer: Model-Based Planning for Web Agents
Code for our paper [*Is Your LLM Secretly a World Model of the Internet*? Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559v1). :construction: :construction: Under Construction

![image](https://github.com/user-attachments/assets/a1189fee-ff43-45fc-a818-3dc6befb6ad2)


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
#### (a) Number of Action Steps

| Steps       | Reactive | Tree Search | WebDreamer |
|-------------|----------|-------------|------------|
| Classifieds | 3.4      | 9.9         | 4.1        |
| Reddit      | 5.1      | 13.6        | 5.2        |
| Shopping    | 4.5      | 11.4        | 4.5        |

#### (b) Task Completion Wall Clock Time

| Seconds     | Reactive | Tree Search | WebDreamer |
|-------------|----------|-------------|------------|
| Classifieds | 68.3     | 74.9        | 183.6      |
| Reddit      | 83.5     | 97.2        | 233.7      |
| Shopping    | 87.7     | 78.7        | 179.4      |

WebDreamer effectively explores the search space through simulations, reducing the reliance on real-world interactions.


## Structure of this repo
[main](https://github.com/OSU-NLP-Group/WebDreamer): Different modules of WebDreamer that can be played with independently.

[vwa](https://github.com/OSU-NLP-Group/WebDreamer/tree/vwa): Code to reproduce our experiments on VisualWebArena.

[mind2web-live](https://github.com/OSU-NLP-Group/WebDreamer/tree/mind2web-live): Code to reproduce our experiments on Mind2Web-live

## WebDreamer Modules Usage

### World Model

```bash
python world_model.py
```

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
