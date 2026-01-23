import torch
import numpy as np
from transformers import LlavaForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image
import json
from tqdm import tqdm
import pickle
import scipy.sparse as sp
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
import nltk
import os
from pycocotools.coco import COCO
from pathlib import Path

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

def load_coco_captions_mapping(coco_ann_file='../data/annotations/captions_val2017.json'):
    """
    Load COCO caption annotations and create a mapping
    
    Returns:
        dict: Mapping from image_id to list of captions
    """
    print(f"Loading COCO annotations from {coco_ann_file}")
    
    if not os.path.exists(coco_ann_file):
        print(f"ERROR: COCO annotation file not found at {coco_ann_file}")
        print(f"Please run the download script first!")
        return {}
    
    coco = COCO(coco_ann_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Create mapping
    id_to_captions = {}
    
    print("Creating caption mapping...")
    for img_id in tqdm(img_ids, desc="Loading captions"):
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        captions = [ann['caption'].strip() for ann in anns if 'caption' in ann]
        id_to_captions[img_id] = captions
    
    print(f"Loaded captions for {len(id_to_captions)} images")
    
    # Show statistics
    caption_counts = [len(caps) for caps in id_to_captions.values()]
    print(f"Captions per image: min={min(caption_counts)}, max={max(caption_counts)}, avg={sum(caption_counts)/len(caption_counts):.1f}")
    
    return id_to_captions

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class GraphExtractor:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """Initialize VLM model and processor"""
        self.device = device
        self.model_name = model_name
        
        if "qwen" in model_name.lower():
            self.model_family = "qwen"
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            # Get the actual text model's hidden dimension
            self.hidden_dim = self.model.config.hidden_size  # Should be 2048 for 3B model
            self.num_layers = self.model.config.num_hidden_layers  # Should be 36 for 3B model
        else:
            self.model_family = "llava"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.num_layers = len(self.model.language_model.model.layers)
            self.hidden_dim = self.model.config.text_config.hidden_size
        
        self.model.eval()
        
        print(f"Model loaded: {model_name}")
        print(f"Model type: {self.model_family}")
        print(f"Layers: {self.num_layers}, Hidden dims: {self.hidden_dim}")

    def extract_hidden_states(self, image, text_prompt=None):
        """Extract hidden states from all layers during forward pass"""
        if self.model_family == "qwen":
            # Proper message format for Qwen2.5-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in one short caption."}
                    ]
                }
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            # Process inputs - images should be a list
            inputs = self.processor(
                text=text,
                images=[image],  # Must be a list for Qwen
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Extract hidden states - they should be directly in outputs
            hidden_states = outputs.hidden_states
            
        else:
            # LLaVA processing
            inputs = self.processor(
                text=text_prompt if text_prompt else "USER: <image>\nCaption: ASSISTANT:",
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            hidden_states = outputs.hidden_states
        
        if hidden_states is None:
            raise RuntimeError(f"hidden_states is None for {self.model_family}")
            
        # Convert to numpy - handle different possible shapes
        hidden_states_np = []
        for h in hidden_states:
            if h.dim() == 3:  # [batch, seq_len, hidden_dim]
                hidden_states_np.append(h[0].detach().cpu().numpy())
            elif h.dim() == 2:  # [seq_len, hidden_dim]
                hidden_states_np.append(h.detach().cpu().numpy())
            else:
                print(f"Warning: Unexpected hidden state shape: {h.shape}")
                hidden_states_np.append(h.squeeze(0).detach().cpu().numpy())
        
        return hidden_states_np

    def compute_correlation_graph(self, hidden_states, sparsity=0.9):
        """Compute correlation matrix from hidden states"""
        neuron_timeseries = hidden_states.T
        correlation_matrix = np.corrcoef(neuron_timeseries)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        if sparsity > 0:
            abs_corr = np.abs(correlation_matrix)
            threshold = np.percentile(abs_corr, sparsity * 100)
            mask = abs_corr > threshold
            correlation_matrix = correlation_matrix * mask
        
        sparse_corr = sp.coo_matrix(correlation_matrix)
        edge_index = np.vstack([sparse_corr.row, sparse_corr.col])
        edge_weight = sparse_corr.data
        
        return {
            'edge_index': edge_index.astype(np.int64),
            'edge_weight': edge_weight.astype(np.float32),
            'num_nodes': self.hidden_dim
        }

    def generate_caption(self, image, max_new_tokens=50):
        """Generate caption for the image"""
        if self.model_family == "qwen":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in one short caption."}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            
            inputs = self.processor(
                text=text,
                images=[image],
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
        else:
            prompt = "USER: <image>\nCaption: ASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,  # Some randomness for better captions
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode - handle batch dimension
        if self.model_family == "qwen":
            generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            # Extract just the caption part after the prompt
            if "caption" in generated_text.lower():
                parts = generated_text.lower().split("caption")
                if len(parts) > 1:
                    caption = parts[-1].strip(". \n")
                else:
                    caption = generated_text.strip()
            else:
                caption = generated_text.strip()
        else:
            generated_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
            if "ASSISTANT:" in generated_text:
                caption = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                caption = generated_text.replace("USER: <image>\nCaption:", "").strip()
        # print(f"Generated caption: {caption}")
        return caption


class MetricsCalculator:
    """Calculate BLEU-4, METEOR, and CIDEr scores"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu4(self, candidate, references):
        """Compute BLEU-4 score (takes max over references)"""
        candidate_tokens = candidate.lower().split()
        reference_tokens = [ref.lower().split() for ref in references]
        
        if not candidate_tokens or not reference_tokens:
            return 0.0
        
        try:
            score = sentence_bleu(
                reference_tokens,
                candidate_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=self.smoothing
            )
            return float(score)
        except:
            return 0.0
    
    def compute_meteor(self, candidate, references):
        """Compute METEOR score (takes max over references)"""
        candidate_tokens = candidate.lower().split()
        
        if not candidate_tokens:
            return 0.0
        
        scores = []
        for ref in references:
            ref_tokens = ref.lower().split()
            if not ref_tokens:
                continue
            try:
                score = meteor_score([ref_tokens], candidate_tokens)
                scores.append(score)
            except:
                continue
        
        # Take max score, not mean
        return float(max(scores)) if scores else 0.0
    
    def compute_cider(self, candidate, references, image_id="0"):
        """Compute CIDEr score (uses consensus of all references)"""
        if not references or len(references) == 0:
            return 0.0
        
        # CIDEr needs multiple diverse references
        if len(set(references)) < 2:
            return 0.0  # Not enough diversity
        
        try:
            # Create a new CIDEr instance
            cider_scorer = Cider()
            
            # Format inputs
            gts = {str(image_id): references}  # All references
            res = {str(image_id): [candidate]}  # Generated caption
            
            # Compute score
            score, _ = cider_scorer.compute_score(gts, res)
            
            if isinstance(score, np.ndarray):
                score = float(score[0]) if len(score) > 0 else 0.0
            else:
                score = float(score) if score is not None else 0.0
            
            return score
            
        except Exception as e:
            print(f"CIDEr error for image {image_id}: {e}")
            return 0.0

def create_dataset(
    num_samples=1000,
    model_name="llava-hf/llava-1.5-7b-hf",
    output_dir="graph_probing_dataset",
    sparsity=0.9,
    verbose=False,
    coco_ann_file='../data/annotations/captions_val2017.json'
):
    """
    Create graph probing dataset for VLM with proper 5 COCO captions
    """
    
    # Determine dataset prefix based on model
    if "qwen" in model_name.lower():
        dataset_prefix = "qwen"
    else:
        dataset_prefix = "llava"
    
    output_dir = f"{dataset_prefix}_{output_dir}_sparsity_{int(sparsity*100)}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO captions mapping FIRST
    print("="*60)
    print("Loading COCO caption annotations...")
    coco_captions = load_coco_captions_mapping(coco_ann_file)
    
    if not coco_captions:
        print("ERROR: Failed to load COCO captions! Exiting...")
        return []
    
    print("="*60)
    print("Loading MS COCO dataset images...")
    dataset = load_dataset("HuggingFaceM4/COCO", split=f"validation[:{num_samples}]", cache_dir='../data/')
    
    print(f"Initializing {model_name} model...")
    extractor = GraphExtractor(model_name=model_name)
    metrics_calc = MetricsCalculator()
    
    layer_indices = {
        'layer_0': 0,
        'layer_middle': extractor.num_layers // 2,
        'layer_last': extractor.num_layers
    }
    
    all_samples = []
    successful_samples = 0
    missing_captions = 0
    
    # Set prompt based on model type
    if extractor.model_family == "qwen":
        graph_extraction_prompt = "<|im_start|>user\n<image>\nCaption:<|im_end|>\n<|im_start|>assistant\n"
    else:
        graph_extraction_prompt = "USER: <image>\nCaption: ASSISTANT:"
    
    for idx, item in enumerate(tqdm(dataset, desc="Processing samples")):
        try:
            image = item['image']
            cocoid = item.get('cocoid', idx)
            
            # Get all 5 captions from COCO annotations
            references = coco_captions.get(cocoid, [])
            
            if not references:
                missing_captions += 1
                if verbose:
                    print(f"Warning: No captions found for COCO ID {cocoid}")
                continue
            
            # Print first sample info
            if successful_samples == 0:
                print(f"\n{'='*60}")
                print(f"First sample info:")
                print(f"  COCO ID: {cocoid}")
                print(f"  Number of captions: {len(references)}")
                for i, ref in enumerate(references):
                    print(f"  Caption {i+1}: {ref[:80]}...")
                print(f"{'='*60}\n")
            
            # Extract hidden states
            hidden_states_all = extractor.extract_hidden_states(image, graph_extraction_prompt)
            # print(f"Extracted hidden states for COCO ID {cocoid}, total layers: {len(hidden_states_all)}")
            # Compute correlation graphs
            graphs = {}
            for layer_name, layer_idx in layer_indices.items():
                print(f"  Processing layer {layer_name} (index {layer_idx})")
                if layer_idx >= len(hidden_states_all):
                    layer_idx = len(hidden_states_all) - 1
                    
                hidden_states = hidden_states_all[layer_idx]
                # print(f"    Hidden states shape: {hidden_states.shape}")
                graph = extractor.compute_correlation_graph(hidden_states, sparsity)
                # print(graph)
                graphs[layer_name] = graph
                # print(f"  Layer {layer_name} (index {layer_idx})")
            
            # Generate caption
            generated_caption = extractor.generate_caption(image)
            print(f"Generated caption for COCO ID {cocoid}: {generated_caption}")
            # Compute metrics with all 5 references
            bleu4 = metrics_calc.compute_bleu4(generated_caption, references)
            meteor = metrics_calc.compute_meteor(generated_caption, references)
            cider = metrics_calc.compute_cider(generated_caption, references, str(cocoid))
            
            sample = {
                'image_id': cocoid,
                'graphs': graphs,
                'metrics': {
                    'bleu4': float(bleu4),
                    'meteor': float(meteor),
                    'cider': float(cider)
                }
            }
            
            all_samples.append(sample)
            successful_samples += 1
            
            # Print details if verbose
            if verbose and successful_samples % 100 == 0:
                print(f"\n{'='*60}")
                print(f"Processed {successful_samples} samples")
                print(f"Latest sample (COCO ID: {cocoid}):")
                print(f"  Generated: {generated_caption[:100]}...")
                print(f"  Metrics: BLEU-4={bleu4:.3f}, METEOR={meteor:.3f}, CIDEr={cider:.3f}")
                print(f"  Missing captions so far: {missing_captions}")
                
        except Exception as e:
            if verbose:
                print(f"Error processing sample {idx}: {str(e)}")
                import traceback
                traceback.print_exc()
            continue
    
    # Calculate final statistics
    if all_samples:
        avg_metrics = {
            'bleu4': np.mean([s['metrics']['bleu4'] for s in all_samples]),
            'meteor': np.mean([s['metrics']['meteor'] for s in all_samples]),
            'cider': np.mean([s['metrics']['cider'] for s in all_samples])
        }
        
        print(f"\n{'='*60}")
        print(f"FINAL STATISTICS")
        print(f"{'='*60}")
        print(f"Samples processed: {len(all_samples)}")
        print(f"Missing captions: {missing_captions}")
        print(f"\nAverage Metrics:")
        print(f"  BLEU-4: {avg_metrics['bleu4']:.4f}")
        print(f"  METEOR: {avg_metrics['meteor']:.4f}")
        print(f"  CIDEr:  {avg_metrics['cider']:.4f}")
        
        # Save complete dataset
        final_path = os.path.join(output_dir, 'complete_dataset.pkl')
        with open(final_path, 'wb') as f:
            pickle.dump(all_samples, f)
        
        # Save metadata
        metadata = {
            'num_samples': len(all_samples),
            'model': model_name,
            'model_family': extractor.model_family,
            'sparsity': sparsity,
            'layers_extracted': list(layer_indices.keys()),
            'metrics': ['bleu4', 'meteor', 'cider'],
            'avg_metrics': avg_metrics,
            'missing_captions': missing_captions
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved to: {output_dir}")
        print(f"Files created:")
        print(f"  - complete_dataset.pkl")
        print(f"  - metadata.json")
        print(f"{'='*60}")
    
    return all_samples

if __name__ == "__main__":
    # Make sure you have the COCO annotations file
    coco_ann_file = '../data/annotations/captions_val2017.json'
    
    if not os.path.exists(coco_ann_file):
        print(f"ERROR: COCO annotations not found at {coco_ann_file}")
        print("Please run the download script first!")
        exit(1)
    
    # Model options
    models = [
        # "llava-hf/llava-1.5-7b-hf",
        "Qwen/Qwen2.5-VL-3B-Instruct"
    ]
    
    sparsity_levels = [0.9, 0.95, 0.99]
    
    for model_name in models:
        for sparsity in sparsity_levels:
            print(f"\n{'='*60}")
            print(f"Creating dataset with model={model_name}, sparsity={sparsity}")
            print(f"{'='*60}")

            name = 'llava' if 'llava' in model_name.lower() else 'qwen'
            
            dataset = create_dataset(
                num_samples=5000,
                model_name=model_name,
                output_dir="graph_probing_dataset",
                sparsity=sparsity,
                verbose=True,
                coco_ann_file=coco_ann_file
            )