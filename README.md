# Visual-Language Model Graph Probing




## requirements


install pytorch 
requirements
follow the download process in here https://github.com/Labbeti/aac-metrics

pip install requirements.txt 


VLM selection:
LLaVA, Qwen-2.5-VL, InternVL3



| Model Name | HF Link | # LLM Layers | Hidden Dimension |
|------------|---------|-------------|------------------|
| llava-hf/llava-1.5-7b-hf           | [link](https://huggingface.co/llava-hf/llava-1.5-7b-hf)           | 32 | 4096  |
| llava-hf/llava-1.5-13b-hf          | [link](https://huggingface.co/llava-hf/llava-1.5-13b-hf)          | 40 | 5120  |
| Qwen/Qwen2.5-VL-3B-Instruct        | [link](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)        | 36 | 2048  |
| Qwen/Qwen2.5-VL-7B-Instruct        | [link](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)        | 28 | 3584  |
| Qwen/Qwen2.5-VL-32B-Instruct       | [link](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct)       | 64 | 5120  |
| OpenGVLab/InternVL3-1B-HF          | [link](https://huggingface.co/OpenGVLab/InternVL3-1B-HF)          | 24 | 896  |
| OpenGVLab/InternVL3-2B-HF          | [link](https://huggingface.co/OpenGVLab/InternVL3-2B-HF)          | 28 | 1536  |
| OpenGVLab/InternVL3-8B-HF          | [link](https://huggingface.co/OpenGVLab/InternVL3-8B-HF)          | 28 | 3584  |
| OpenGVLab/InternVL3-14B-HF         | [link](https://huggingface.co/OpenGVLab/InternVL3-14B-hf)         | 48 | 5120  |





#### TUIDC


```bash
# QA files (~70 MB)
wget -O TDIUC.zip https://kushalkafle.com/data/TDIUC.zip
unzip TDIUC.zip 

# coco 2014 val (~6GB)
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```


Both TDIUC and val2014 should be available inside your repository.

