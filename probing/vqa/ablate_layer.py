import torch
import numpy as np
from datasets import load_dataset
from PIL import Image
import json
from tqdm import tqdm
import os
import argparse
from typing import List, Tuple, Optional
import torch.nn.functional as F

from utils import *
from model import NeuronGraphExtractor as GraphExtractor

def split_clevr_question_answer(qa_str):
    list_ = qa_str.split("?")
    if len(list_) != 2:
        raise ValueError(f"Unexpected QA format, cannot split question and answer: {qa_str}")
    question = list_[0] + "?"
    answer = list_[1].strip().lower()
    return question, answer

CLEVR_COLORS = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
CLEVR_SHAPES = ['cube', 'sphere', 'cylinder']

def constrain_clevr_prompt(question: str, task: str) -> str:
    if task == "color":
        choices = ", ".join(CLEVR_COLORS)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    elif task == "counting":
        return f"{question} Answer with a single integer 0-10. Output only the number."
    elif task == "existence":
        return f"{question} Answer with 'yes' or 'no' only. Output exactly one word."
    elif task == "comparison":
        return f"{question} Answer with 'more', 'fewer', or 'equal' only. Output exactly one word."
    elif task == "shape":
        choices = ", ".join(CLEVR_SHAPES)
        return f"{question} Answer with one word from: {choices}. Output exactly one word."
    else:
        return question

def classify_clevr_question(question, answer):
    question = question.lower()
    answer = answer.lower()
    try:
        num = int(answer)
        if 0 <= num <= 10:
            return 'counting'
    except:
        pass
    if "color" in question and answer in CLEVR_COLORS:
        return 'color'
    elif answer in CLEVR_SHAPES:
        return 'shape'
    else:
        return 'unknown'


def select_neurons_to_ablate(
    method: str,
    layer_idx: int,
    ablation_percentage: float,
    degree_data: np.ndarray,
    activation_data: np.ndarray,
    hidden_dim: int
) -> np.ndarray:
    """
    Select which neurons to ablate based on the method.
    
    Args:
        method: 'random', 'top_degree', or 'top_activation'
        layer_idx: which layer to ablate
        ablation_percentage: fraction of neurons to ablate (e.g., 0.1 for 10%)
        degree_data: [num_layers, hidden_dim] average node degrees
        activation_data: [num_layers, hidden_dim] average activations
        hidden_dim: dimension of hidden states
    
    Returns:
        indices: np.ndarray of neuron indices to ablate
    """
    num_to_ablate = int(ablation_percentage * hidden_dim)
    
    if method == 'random':
        indices = np.random.choice(hidden_dim, size=num_to_ablate, replace=False)
    elif method == 'top_degree':
        layer_degrees = degree_data[layer_idx]
        indices = np.argsort(layer_degrees)[-num_to_ablate:]
    elif method == 'top_activation':
        layer_activations = activation_data[layer_idx]
        indices = np.argsort(layer_activations)[-num_to_ablate:]
    else:
        raise ValueError(f"Unknown ablation method: {method}")
    
    return indices


class AblationGraphExtractor(GraphExtractor):
    """Extended GraphExtractor that supports neuron ablation during generation."""
    
    def __init__(self, *args, ablation_layer: int = None, ablation_indices: np.ndarray = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_layer = ablation_layer
        self.ablation_indices = ablation_indices
        self.hooks = []
        
    def ablation_hook(self, module, input, output):
        """Hook to zero out neurons during forward pass."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Zero out the ablated neurons
        if self.ablation_indices is not None and len(self.ablation_indices) > 0:
            hidden_states[:, :, self.ablation_indices] = 0.0
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states
    
    def register_ablation_hooks(self):
        """Register forward hooks for ablation."""
        if self.ablation_layer is None or self.ablation_indices is None:
            return
        
        target_layer = self.model.model.layers[self.ablation_layer]
        hook = target_layer.register_forward_hook(self.ablation_hook)
        self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def compute_answer_probability(self, outputs, answer: str, task: str) -> Optional[float]:
        """
        Compute probability of correct answer from generation outputs.
        
        For single-token answers (color, shape, existence, comparison): 
            Use first token probability
        For multi-token answers (counting):
            Compute joint probability across all tokens, or return None
        """
        if len(outputs.scores) == 0:
            return None
        
        # For single-word answers (color, shape, existence, comparison)
        if task in ['color', 'shape', 'existence', 'comparison']:
            # Get first token distribution
            first_token_logits = outputs.scores[0][0]  # [vocab_size]
            probs = F.softmax(first_token_logits, dim=-1)
            
            # Get token IDs for the answer
            answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            
            # Return max probability among possible tokenizations
            answer_probs = [probs[tid].item() for tid in answer_tokens]
            return max(answer_probs) if answer_probs else None
        
        # For counting task - multi-token answer possible
        elif task == 'counting':
            # Tokenize the answer to see how many tokens it needs
            answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
            
            if len(answer_tokens) == 1:
                # Single token answer (e.g., "0", "1", ...)
                first_token_logits = outputs.scores[0][0]
                probs = F.softmax(first_token_logits, dim=-1)
                return probs[answer_tokens[0]].item()
            else:
                # Multi-token answer - we could compute joint probability
                # but this gets complicated with autoregressive generation
                # For now, return None for multi-token counting answers
                return None
        
        return None
    
    def process_single_with_ablation(self, image, prompt: str, answer: str, task: str):
        """
        Process single sample with ablation and return probability of correct answer.
        """
        self.register_ablation_hooks()
        
        try:
            formatted_prompt = prompt
            inputs = self.processor(
                text=formatted_prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            generated_ids = outputs.sequences
            gen_text = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            # BETTER APPROACH: Get probability of the ACTUALLY GENERATED token
            correct_prob = None
            if len(outputs.scores) > 0:
                first_token_logits = outputs.scores[0][0]  # [vocab_size]
                probs = F.softmax(first_token_logits, dim=-1)
                
                # Get the token that was actually generated
                generated_token_id = outputs.sequences[0, inputs['input_ids'].shape[1]].item()
                generated_token_prob = probs[generated_token_id].item()
                
                # Check if generation matches answer
                is_correct = gen_text.lower().strip() == answer
                
                # If correct, use the generated token's probability
                # If incorrect, try to find the correct answer's probability
                if is_correct:
                    correct_prob = generated_token_prob
                else:
                    # Try to get probability of what the correct answer would have been
                    answer_tokens = self.processor.tokenizer.encode(answer, add_special_tokens=False)
                    if answer_tokens:
                        answer_probs = [probs[tid].item() for tid in answer_tokens]
                        correct_prob = max(answer_probs) if answer_probs else None
            
            return gen_text, correct_prob
            
        finally:
            self.remove_hooks()


def run_clevr_ablation_experiment(
    model_name: str = "llava-hf/llava-1.5-7b-hf",
    task: str = "color",
    ablation_layer: int = 15,
    ablation_method: str = "random",
    ablation_percentage: float = 0.1,
    num_samples: int = 500,
    sparse_level: float = 0.9,
    data_dir: str = None,
    device: str = "cuda:0",
    log_every: int = 10,
):
    """Run ablation experiment on CLEVR dataset with probability tracking."""
    
    model_prefix = model_path2name(model_name)
    
    if data_dir is None:
        data_dir = f"data/{model_prefix}_clevr_{task}_sparsity_{int(sparse_level * 100)}_layer_analysis"
    
    print("="*60)
    print("ABLATION EXPERIMENT SETUP")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Task: {task}")
    print(f"Ablation layer: {ablation_layer}")
    print(f"Ablation method: {ablation_method}")
    print(f"Ablation percentage: {ablation_percentage*100:.1f}%")
    print(f"Data directory: {data_dir}")
    
    if task == 'counting':
        print("\nNote: Counting task may have multi-token answers.")
        print("Probability will be None for multi-token answers (e.g., '10').")
    
    print("="*60)
    
    # Load layer analysis data
    degree_path = os.path.join(data_dir, "avg_node_degree_all_layers.npy")
    activation_path = os.path.join(data_dir, "avg_last_token_activation_all_layers.npy")
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    if not os.path.exists(degree_path):
        raise FileNotFoundError(f"Degree data not found: {degree_path}")
    if not os.path.exists(activation_path):
        raise FileNotFoundError(f"Activation data not found: {activation_path}")
    
    degree_data = np.load(degree_path)
    activation_data = np.load(activation_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    hidden_dim = metadata['hidden_dim']
    num_layers = metadata['num_layers']
    
    print(f"\nLoaded layer analysis data:")
    print(f"  Shape: {degree_data.shape}")
    print(f"  Layers: {num_layers}, Hidden dim: {hidden_dim}")
    
    if ablation_layer < 0 or ablation_layer >= num_layers:
        raise ValueError(f"Invalid ablation layer {ablation_layer}, must be in [0, {num_layers-1}]")
    
    # Select neurons to ablate
    ablation_indices = select_neurons_to_ablate(
        method=ablation_method,
        layer_idx=ablation_layer,
        ablation_percentage=ablation_percentage,
        degree_data=degree_data,
        activation_data=activation_data,
        hidden_dim=hidden_dim
    )
    
    print(f"\nSelected {len(ablation_indices)} neurons to ablate at layer {ablation_layer}")
    print(f"Ablation indices (first 20): {ablation_indices[:20]}")
    
    if ablation_method == 'top_degree':
        selected_degrees = degree_data[ablation_layer, ablation_indices]
        print(f"Degree range of selected neurons: [{selected_degrees.min():.2f}, {selected_degrees.max():.2f}]")
    elif ablation_method == 'top_activation':
        selected_activations = activation_data[ablation_layer, ablation_indices]
        print(f"Activation range: [{selected_activations.min():.4f}, {selected_activations.max():.4f}]")
    
    print("="*60)
    
    # Load CLEVR dataset
    print("\nLoading CLEVR dataset...")
    clevr_samples = load_dataset("laion/clevr-webdataset", split='validation')
    
    valid_sample_counts = 0
    all_images, all_questions, all_answers = [], [], []
    for sample in clevr_samples:
        txt = sample['txt']
        try:
            question, answer = split_clevr_question_answer(txt)
            if classify_clevr_question(question, answer) == task:
                valid_sample_counts += 1
                all_questions.append(question)
                all_answers.append(answer)
                all_images.append(sample['jpg'].convert("RGB"))
        except Exception:
            continue
        if valid_sample_counts >= num_samples:
            break
    
    print(f"Loaded {valid_sample_counts} samples for task={task}")
    
    # Initialize model with ablation
    print(f"\nInitializing model with ablation...")
    extractor = AblationGraphExtractor(
        model_name=model_name,
        device=device,
        ablation_layer=ablation_layer,
        ablation_indices=ablation_indices
    )
    
    # Run with ablation
    print("\n" + "="*60)
    print("RUNNING WITH ABLATION")
    print("="*60)
    
    ablation_correct = 0
    ablation_total = 0
    correct_answer_probs = []
    per_sample_results = []
    
    for sample_idx in tqdm(range(valid_sample_counts), desc="Ablation experiment"):
        image = all_images[sample_idx]
        question = all_questions[sample_idx]
        answer = all_answers[sample_idx]
        
        try:
            prompt = constrain_clevr_prompt(question, task)
            prompt = prompt_for_model(prompt, model_type=extractor.model_type)
            
            # Process with ablation
            gen, correct_prob = extractor.process_single_with_ablation(
                image, prompt, answer, task
            )
            
            is_correct = isinstance(gen, str) and gen.lower().strip() == answer
            if is_correct:
                ablation_correct += 1
            ablation_total += 1
            
            if correct_prob is not None:
                correct_answer_probs.append(correct_prob)
            
            per_sample_results.append({
                "sample_idx": sample_idx,
                "question": question,
                "answer": answer,
                "prediction": gen,
                "correct": is_correct,
                "correct_answer_probability": correct_prob
            })
            
            if sample_idx % log_every == 0 and sample_idx > 0:
                print(f"\n--- Sample {sample_idx} ---")
                print(f"Q: {question}")
                print(f"Pred: {gen} | Ref: {answer} | Correct: {is_correct}")
                if correct_prob is not None:
                    print(f"P(correct answer): {correct_prob:.4f}")
                else:
                    print(f"P(correct answer): N/A (multi-token)")
            
        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        torch.cuda.empty_cache()
    
    ablation_accuracy = ablation_correct / ablation_total if ablation_total > 0 else 0.0
    avg_correct_prob = np.mean(correct_answer_probs) if correct_answer_probs else None
    
    # Save results
    output_dir = f"data/{model_prefix}_clevr_{task}_ablation_layer{ablation_layer}_{ablation_method}_{int(ablation_percentage*100)}pct"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {
        "model": model_name,
        "task": task,
        "ablation_layer": ablation_layer,
        "ablation_method": ablation_method,
        "ablation_percentage": ablation_percentage,
        "num_neurons_ablated": len(ablation_indices),
        "ablation_indices": ablation_indices.tolist(),
        "ablation_accuracy": ablation_accuracy,
        "ablation_correct": ablation_correct,
        "ablation_total": ablation_total,
        "avg_correct_answer_probability": avg_correct_prob,
        "num_samples_with_probability": len(correct_answer_probs),
    }
    
    with open(os.path.join(output_dir, "ablation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(output_dir, "per_sample_results.json"), 'w') as f:
        json.dump(per_sample_results, f, indent=2)
    
    np.save(os.path.join(output_dir, "ablation_indices.npy"), ablation_indices)
    if correct_answer_probs:
        np.save(os.path.join(output_dir, "correct_answer_probabilities.npy"), 
                np.array(correct_answer_probs))
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Ablation Accuracy: {ablation_accuracy*100:.2f}% ({ablation_correct}/{ablation_total})")
    if avg_correct_prob is not None:
        print(f"Avg P(correct answer): {avg_correct_prob:.4f}")
        print(f"Samples with probability: {len(correct_answer_probs)}/{ablation_total}")
    else:
        print(f"No probability computed (all answers were multi-token)")
    print(f"\nResults saved to: {output_dir}")
    print("="*60)
    return ablation_accuracy, avg_correct_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neuron ablation experiment on CLEVR")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--task", type=str, default="color", 
                        choices=['color', 'counting', 'existence', 'comparison', 'shape'])
    parser.add_argument("--ablation_layer", type=int, default=15)
    parser.add_argument("--ablation_method", type=str, default="random",
                        choices=['random', 'top_degree', 'top_activation'])
    parser.add_argument("--ablation_percentage", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--sparse_level", type=float, default=0.9)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    
    run_clevr_ablation_experiment(
        model_name=args.model_name,
        task=args.task,
        ablation_layer=args.ablation_layer,
        ablation_method=args.ablation_method,
        ablation_percentage=args.ablation_percentage,
        num_samples=args.num_samples,
        sparse_level=args.sparse_level,
        data_dir=args.data_dir,
        device=args.device,
        log_every=args.log_every,
    )