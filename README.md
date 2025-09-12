# Visual-Language Model Graph Probing



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
| Salesforce/blip2-opt-2.7b          | [link](https://huggingface.co/Salesforce/blip2-opt-2.7b)                  | 32           | 2560             |
| Salesforce/blip2-opt-6.7b          | [link](https://huggingface.co/Salesforce/blip2-opt-6.7b)                  | 32           | 4096             |
| Salesforce/blip2-flan-t5-xxl       | [link](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)               | 24           | 4096             |

Qwen needs a chat template but LLaVA and BLIP2 just build raw prompt