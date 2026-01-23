import torch
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import pickle
import scipy.sparse as sp
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
import os

from utils import model_path2name, caption_prompt


class GraphExtractor:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """Initialize VLM model and processor"""
        self.device = device
        self.model_name = model_name
        
        # Determine model type and load accordingly
        if model_name.lower().startswith("qwen"):
            self.model_family = "qwen"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model.config.output_hidden_states = True
        else:
            self.model_family = "llava"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model.eval()
        
        # Get model configuration
        if self.model_family == "qwen":
            # Adjust for Qwen model structure
            self.num_layers = len(self.model.model.layers) if hasattr(self.model, 'model') else 24
            self.hidden_dim = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else 2048
        else:
            self.num_layers = len(self.model.language_model.model.layers)
            self.hidden_dim = self.model.config.text_config.hidden_size
        
        print(f"Model loaded: {model_name}")
        print(f"Model type: {self.model_family}")
        print(f"Layers: {self.num_layers}, Hidden dims: {self.hidden_dim}")
    
    def extract_hidden_states(self, image, text_prompt="USER: <image>\nCaption: ASSISTANT:"):
        """Extract hidden states from all layers during forward pass"""
        
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states
        hidden_states_np = [h[0].cpu().numpy() for h in hidden_states]
        
        return hidden_states_np
    
    def compute_correlation_graph(self, hidden_states):
        """Compute correlation matrix from hidden states"""
        neuron_timeseries = hidden_states.T
        correlation_matrix = np.corrcoef(neuron_timeseries)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        sparse_corr = sp.coo_matrix(correlation_matrix)
        edge_index = np.vstack([sparse_corr.row, sparse_corr.col])
        edge_weight = sparse_corr.data
        
        return {
            'edge_index': edge_index.astype(np.int64),
            'edge_weight': edge_weight.astype(np.float16),
            'num_nodes': self.hidden_dim
        }
    
    def generate_caption(self, image, max_new_tokens=50):
        """Generate caption for the image"""
        if self.model_family == "qwen":
            prompt = "<|im_start|>user\n<image>\nProvide a caption for this image.<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = "USER: <image>\nCaption: ASSISTANT:"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                top_p=0.95,
                num_beams=1,
                pad_token_id=self.processor.tokenizer.pad_token_id
                    if self.processor.tokenizer.pad_token_id is not None
                    else self.processor.tokenizer.eos_token_id,
            )
        
        generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract caption based on model type
        if self.model_family == "qwen":
            if "assistant" in generated_text:
                caption = generated_text.split("assistant")[-1].strip()
            else:
                caption = generated_text.replace(prompt, "").strip()
        else:
            if "ASSISTANT:" in generated_text:
                caption = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                caption = generated_text.replace(prompt, "").strip()
        
        return caption


def create_dataset(
    num_samples=1000,
    model_name="llava-hf/llava-1.5-7b-hf",
    output_dir="probing_dataset",
    prompt_choice=0,
    verbose=False,
    use_hf=True,
    device="cuda:1"
):
    """
    Create graph probing dataset for VLM with proper COCO captions
    """
    
    dataset_prefix = model_path2name(model_name)
    
    output_dir = f"data/{dataset_prefix}_prompt_{prompt_choice}_{output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO dataset from HuggingFace
    print("="*60)
    print("Loading COCO dataset from HuggingFace...")
    coco_samples = load_dataset("lmms-lab/COCO-Caption2017", split='val')
    
    if not coco_samples:
        print("ERROR: Failed to load COCO dataset from HuggingFace!")
        return []
    
    print(f"Loaded {len(coco_samples)} samples from COCO 2017 validation set")
    print("="*60)
    
    print(f"Initializing {model_name} model...")
    extractor = GraphExtractor(model_name=model_name, device=device)
    
    layer_indices = {
        'layer_0': 0,
        'layer_middle': extractor.num_layers // 2,
        'layer_last': extractor.num_layers
    }
    
    all_samples = []
    gts, res = {}, {}
    successful_samples = 0
    missing_captions = 0
    graph_extraction_prompt = caption_prompt(prompt_choice)
    print(f"Using graph extraction prompt:\n{graph_extraction_prompt}")
    
    for idx, item in enumerate(tqdm(coco_samples, desc="Processing samples")):
        try:
            image = item['image']
            if not isinstance(image, Image.Image):
                print(f"Warning: Sample {idx} doesn't have a valid image")
                continue
                
            # Ensure image is in RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            cocoid = item['question_id']
            references = item['answer']
            
            # Print first sample info
            if idx == 0:
                print(f"\n{'='*60}")
                print(f"First sample info:")
                # print(f"  Image ID: {cocoid}")
                print(f"  Number of captions: {len(references)}")
                for i, ref in enumerate(references[:3]):  # Show first 3 captions
                    print(f"  Caption {i+1}: {ref[:80]}...")
                print(f"{'='*60}\n")
            
            # Extract hidden states
            hidden_states_all = extractor.extract_hidden_states(image, graph_extraction_prompt)
            
            # Compute correlation graphs for three layers
            graphs = {}
            for layer_name, layer_idx in layer_indices.items():
                if layer_idx >= len(hidden_states_all):
                    layer_idx = len(hidden_states_all) - 1
                    
                hidden_states = hidden_states_all[layer_idx]
                graph = extractor.compute_correlation_graph(hidden_states)
                graphs[layer_name] = graph
            
            # Generate caption
            generated_caption = extractor.generate_caption(image)
            
            if not generated_caption:
                missing_captions += 1

            # Store for CIDEr calculation
            gts[idx] = references
            res[idx] = [generated_caption]
            print(f"Sample {idx} - Generated caption: {generated_caption[:100]}...")
            
            last_token_state = hidden_states_all[-1][-1, :]  # Exclude special tokens
            
            # Create sample dictionary with all required information
            sample = {
                'references': references,
                'generated_caption': generated_caption,
                'graph_layer_0': graphs['layer_0'],
                'graph_layer_middle': graphs['layer_middle'],
                'graph_layer_last': graphs['layer_last'],
                'last_token_state': last_token_state,
            }
            
            all_samples.append(sample)
            successful_samples += 1
            
            # Print details if verbose
            if verbose and successful_samples % 100 == 0:
                print(f"\n{'='*60}")
                print(f"Processed {successful_samples} samples")
                print(f"Latest sample (Image ID: {cocoid}):")
                print(f"  Generated: {generated_caption[:100]}...")
                print(f"  Last hidden state shape: {last_token_state.shape}")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate CIDEr scores
    cider_scores_per_sample = {}
    overall_cider = 0.0
    
    if gts and res:
        cider_scorer = Cider()
        overall_cider, individual_scores = cider_scorer.compute_score(gts, res)
        
        # Map individual scores back to samples
        for i, score in enumerate(individual_scores):
            cider_scores_per_sample[i] = score
        
        # Add CIDEr score to each sample
        for i, sample in enumerate(all_samples):
            sample['cider_score'] = cider_scores_per_sample.get(i, 0.0)
    
    # Calculate final statistics
    if all_samples:
        print(f"\n{'='*60}")
        print(f"FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Samples processed: {len(all_samples)}")
        print(f"Missing captions: {missing_captions}")
        print(f"\nAverage Metrics:")
        print(f"  Overall CIDEr:  {overall_cider:.4f}")
        
        # Verify data structure
        sample_check = all_samples[0]
        print(f"\nSample structure verification:")
        print(f"  Keys in sample: {list(sample_check.keys())}")
        print(f"  Graph layer 0 keys: {list(sample_check['graph_layer_0'].keys())}")
        print(f"  Last hidden state shape: {sample_check['last_token_state'].shape}")
        print(f"  Individual CIDEr score: {sample_check['cider_score']:.4f}")
        
        # Save complete dataset with all information
        final_path = os.path.join(output_dir, 'complete_dataset.pkl')
        with open(final_path, 'wb') as f:
            pickle.dump(all_samples, f)
        
        # Save metadata
        metadata = {
            'num_samples': len(all_samples),
            'model': model_name,
            'model_family': extractor.model_family,
            'num_layers': extractor.num_layers,
            'hidden_dim': extractor.hidden_dim,
            'layers_extracted': list(layer_indices.keys()),
            'layer_indices': layer_indices,
            'data_source': 'HuggingFace COCO dataset',
            'prompt_choice': prompt_choice,
            'prompt_used': graph_extraction_prompt,
            'overall_cider_score': overall_cider,
            'sample_keys': list(all_samples[0].keys()) if all_samples else []
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to: {output_dir}")
        print(f"Files created:")
        print(f"  - complete_dataset.pkl (contains all samples with graphs, hidden states, and CIDEr scores)")
        print(f"  - metadata.json")
        print(f"{'='*60}")
    
    return all_samples


if __name__ == "__main__":
    # Model options
    models = [
        "llava-hf/llava-1.5-7b-hf", 
        # "Qwen/Qwen2-VL-2B-Instruct",
        # "llava-hf/llava-interleave-qwen-0.5b-hf",
        # "bczhou/tiny-llava-v1-hf",
    ]
    
    prompt_choices = [2]
    
    for model_name in models:
        for prompt_choice in prompt_choices:
            print(f"\n{'='*60}")
            print(f"Creating dataset with model={model_name}, prompt={prompt_choice}")
            print(f"{'='*60}")
            
            dataset = create_dataset(
                num_samples=500,  
                model_name=model_name,
                output_dir="probing_dataset",
                prompt_choice=prompt_choice,
                verbose=False,
                use_hf=True,  # Use HuggingFace datasets
                device="cuda:1"
            )