# Structural Graph Probing of Vision–Language Models

Code for the paper [Structural Graph Probing of Vision–Language Models](https://arxiv.org/abs/2603.27070).

Overview

## Repository Layout

**Core modules:**

- `model.py` — model loading, hidden-state extraction, correlation graph construction
- `dataset.py` — dataset loading and VQA prompt construction
- `utils.py` — shared helpers, constants, and text sanitization utilities

**Experiment scripts** (under `probing/`):


| Script                | Description                                                               |
| --------------------- | ------------------------------------------------------------------------- |
| `extract_graphs.py`   | Extract hidden states, build correlation graphs, save per-layer artifacts |
| `degree_analysis.py`  | Layer-wise degree and activation analysis across all layers               |
| `hub_neurons.py`      | Identify hub neurons by degree and activation frequency                   |
| `intervene_neuron.py` | Neuron-level ablation / scaling intervention experiments                  |
| `intervene_edge.py`   | Edge-level (neuron-pair) intervention experiments                         |
| `modality_corr.py`    | Cross-modality (visual-text) token correlation analysis                   |


**Scripts:**

- `scripts/examples/` — ready-to-run example scripts for each pipeline step (start here)
- `scripts/verify/` — smoke tests for quick verification

## Environment Setup

```bash
conda create -n vlm-graph-probing python=3.10
conda activate vlm-graph-probing
pip install -r requirements.txt
```

**Notes:**

- A working PyTorch installation compatible with your CUDA setup is required for GPU experiments.
- Some Hugging Face model checkpoints may require authentication or acceptance of model terms.

## Supported Models


| Model         | Hugging Face                                                                      |
| ------------- | --------------------------------------------------------------------------------- |
| LLaVA-1.5-7B  | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)       |
| Qwen2.5-VL-3B | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| InternVL3-1B  | [OpenGVLab/InternVL3-1B-hf](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)     |


See `utils.py` for the full list of supported checkpoints.

## Data Setup


| Dataset           | Source                                                         | Setup                                                                                            |
| ----------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **CLEVR**         | HuggingFace                                                    | Loaded automatically via `datasets` (`laion/clevr-webdataset`). No manual download needed.       |
| **COCO Captions** | HuggingFace                                                    | Loaded automatically via `datasets` (`lmms-lab/COCO-Caption2017`). No manual download needed.    |
| **TDIUC**         | [Manual download](https://kushalkafle.com/projects/tdiuc.html) | Download and place under `data/TDIUC/`. Also requires COCO val2014 images under `data/val2014/`. |


For TDIUC, the expected directory structure is:

```
data/
├── TDIUC/
│   ├── Questions/
│   │   └── OpenEnded_mscoco_val2014_questions.json
│   └── Annotations/
│       └── mscoco_val2014_annotations.json
└── val2014/
    └── COCO_val2014_*.jpg
```

You can override dataset locations via environment variables or CLI flags:

```bash
export VLM_GRAPH_DATA_ROOT=/path/to/data
export VLM_GRAPH_TDIUC_ROOT=/path/to/data/TDIUC
export VLM_GRAPH_COCO_VAL_ROOT=/path/to/data/val2014
```

## Quickstart

Ready-to-run example scripts are provided under `scripts/`. Each script has configurable variables (model, dataset, device, etc.) at the top. Run them in order:

```bash
# Step 1: Extract hidden states and build correlation graphs
bash scripts/01_extract_graphs.sh

# Step 2: Layer-wise degree and activation analysis
bash scripts/02_degree_analysis.sh

# Step 3: Identify hub neurons
bash scripts/03_hub_neurons.sh

# Step 4: Neuron-level intervention (requires step 3 output)
bash scripts/04_intervene_neuron.sh

# Step 5: Edge-level intervention
bash scripts/05_intervene_edge.sh

# Step 6: Cross-modality correlation analysis
bash scripts/06_modality_corr.sh
```

All scripts default to **InternVL3-1B** on **CLEVR color** with `cuda:0`. Edit the variables at the top of each script to change the model, dataset, category, or GPU device.

## Citation

If you use this repository, please consider cite our paper.

```bibtex
@article{he2026structural,
  title={Structural Graph Probing of Vision-Language Models},
  author={He, Haoyu and Zhuo, Yue and Zheng, Yu and Wang, Qi R},
  journal={arXiv preprint arXiv:2603.27070},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.