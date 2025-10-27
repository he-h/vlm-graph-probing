# Visual-Language Model Graph Probing




## requirements


install pytorch 
requirements
MS-coco 
follow the download process in here https://github.com/Labbeti/aac-metrics




VLM selection:
LLaVA, BLIP-2, Qwen-2.5-VL

list of LLaVA


| Model Name                         | HF Link                                                                 | # LLM Layers | Hidden Dimension |
|-----------------------------------|--------------------------------------------------------------------------|--------------|------------------|
| llava-hf/llava-1.5-7b-hf           | [link](https://huggingface.co/llava-hf/llava-1.5-7b-hf)                   | 32            | 4096                |
| llava-hf/llava-1.5-13b-hf          | [link](https://huggingface.co/llava-hf/llava-1.5-13b-hf)                  | 40            | 5120                |
| Qwen/Qwen2.5-VL-3B-Instruct        | [link](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)               | 36           | 2048             |
| Qwen/Qwen2.5-VL-7B-Instruct        | [link](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).              | 28            | 3584                |
| Qwen/Qwen2.5-VL-32B-Instruct       | [link](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)               | 64           | 5120             |

Qwen needs a chat template but LLaVA and BLIP2 just build raw prompt