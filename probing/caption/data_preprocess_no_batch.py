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

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

def load_coco_from_hf(split='validation', num_samples=5000):
    """
    Load COCO dataset directly from Hugging Face
    
    Returns:
        list: List of dicts with 'image', 'image_id', and 'captions'
    """
    print(f"Loading COCO 2017 {split} dataset from Hugging Face...")
    
    # Load COCO captions dataset from HF
    dataset = load_dataset("HuggingFaceM4/COCO", split=split, streaming=False)
    
    items = []
    
    # Convert to list format we need
    for idx, sample in enumerate(tqdm(dataset, desc="Loading COCO samples", total=num_samples)):
        if idx >= num_samples:
            break
            
        # Each sample has 'image' (PIL Image) and 'sentences' (list of captions)
        items.append({
            'image': sample['image'],
            'image_id': sample['image_id'] if 'image_id' in sample else idx,
            'captions': [sent['raw'] for sent in sample['sentences']] if 'sentences' in sample else []
        })
    
    print(f"Loaded {len(items)} samples from COCO")
    
    # Show statistics
    caption_counts = [len(item['captions']) for item in items]
    if caption_counts:
        print(f"Captions per image: min={min(caption_counts)}, max={max(caption_counts)}, avg={sum(caption_counts)/len(caption_counts):.1f}")
    
    return items

class GraphExtractor:
    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """Initialize VLM model and processor"""
        self.device = device
        self.model_name = model_name
        
        # Determine model type and load accordingly
        if "qwen" in model_name.lower():
            self.model_family = "qwen"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model.config.output_hidden_states = True
        else:
            self.model_family = "llava"
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
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
        # Adjust prompt for Qwen if needed
        if self.model_family == "qwen":
            text_prompt = "<|im_start|>user\n<image>\nProvide a caption for this image.<|im_end|>\n<|im_start|>assistant\n"
        
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
            prompt = "<|im_start|>user\n<image>\nProvide a caption for this image.<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = "USER: <image>\nCaption: ASSISTANT:"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                num_beams=1,
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
    use_hf=True  # Flag to use HuggingFace datasets
):
    """
    Create graph probing dataset for VLM with proper COCO captions
    """
    
    # Determine dataset prefix based on model
    if "qwen" in model_name.lower():
        dataset_prefix = "qwen"
    else:
        dataset_prefix = "llava"
    
    output_dir = f"{dataset_prefix}_{output_dir}_sparsity_{int(sparsity*100)}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO dataset from HuggingFace
    print("="*60)
    print("Loading COCO dataset from HuggingFace...")
    coco_samples = load_coco_from_hf(split='validation', num_samples=num_samples)
    
    if not coco_samples:
        print("ERROR: Failed to load COCO dataset from HuggingFace!")
        return []
    
    print(f"Loaded {len(coco_samples)} samples from COCO 2017 validation set")
    print("="*60)
    
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
    
    for idx, item in enumerate(tqdm(coco_samples, desc="Processing samples")):
        try:
            image = item['image']
            if not isinstance(image, Image.Image):
                print(f"Warning: Sample {idx} doesn't have a valid image")
                continue
                
            # Ensure image is in RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            cocoid = item['image_id']
            references = item['captions']
            
            if not references:
                missing_captions += 1
                if verbose:
                    print(f"Warning: No captions found for image {cocoid}")
                continue
            
            # Print first sample info
            if successful_samples == 0:
                print(f"\n{'='*60}")
                print(f"First sample info:")
                print(f"  Image ID: {cocoid}")
                print(f"  Number of captions: {len(references)}")
                for i, ref in enumerate(references[:3]):  # Show first 3 captions
                    print(f"  Caption {i+1}: {ref[:80]}...")
                print(f"{'='*60}\n")
            
            # Extract hidden states
            hidden_states_all = extractor.extract_hidden_states(image, graph_extraction_prompt)
            
            # Compute correlation graphs
            graphs = {}
            for layer_name, layer_idx in layer_indices.items():
                if layer_idx >= len(hidden_states_all):
                    layer_idx = len(hidden_states_all) - 1
                    
                hidden_states = hidden_states_all[layer_idx]
                graph = extractor.compute_correlation_graph(hidden_states, sparsity)
                graphs[layer_name] = graph
            
            # Generate caption
            generated_caption = extractor.generate_caption(image)
            
            # Compute metrics with all references
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
                print(f"Latest sample (Image ID: {cocoid}):")
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
            'missing_captions': missing_captions,
            'data_source': 'HuggingFace COCO dataset'
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
    # Model options
    models = [
        "llava-hf/llava-1.5-7b-hf",
        # "Qwen/Qwen2.5-VL-3B-Instruct"
    ]
    
    sparsity_levels = [0.9, 0.95, 0.99]
    
    for model_name in models:
        for sparsity in sparsity_levels:
            print(f"\n{'='*60}")
            print(f"Creating dataset with model={model_name}, sparsity={sparsity}")
            print(f"{'='*60}")
            
            dataset = create_dataset(
                num_samples=5000,  # Will load 5000 samples from HF
                model_name=model_name,
                output_dir="graph_probing_dataset",
                sparsity=sparsity,
                verbose=False,
                use_hf=True  # Use HuggingFace datasets
            )