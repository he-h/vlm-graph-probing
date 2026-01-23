# import json
# import matplotlib.pyplot as plt
# import numpy as np

# # path1 = "results/intervention/intervention_InternVL3-1B_tdiuc_color_full_degree_top8_scale0.0_all_layers.json"
# # path2 = "results/intervention/intervention_InternVL3-1B_tdiuc_color_last_token_top8_scale0.0_all_layers.json"

# path1 = "results/intervention/intervention_InternVL3-1B_tdiuc_counting_full_degree_top8_scale0.0_all_layers.json"
# path2 = "results/intervention/intervention_InternVL3-1B_tdiuc_counting_last_token_top8_scale0.0_all_layers.json"

# # path1 = "results/intervention/intervention_Qwen2.5-VL-3B_tdiuc_counting_full_degree_top20_scale0.0_all_layers.json"
# # path2 = "results/intervention/intervention_Qwen2.5-VL-3B_tdiuc_counting_last_token_top20_scale0.0_all_layers.json"

# # Load data
# with open(path1, 'r') as f:
#     data1 = json.load(f)
    
# with open(path2, 'r') as f:
#     data2 = json.load(f)

# # Extract layer information
# selected_layers = data1['experiment_config']['selected_layers']
# layer_list = ["layer_" + str(i) for i in selected_layers]

# # Extract accuracies
# acc_full_degree = []
# acc_last_token = []

# for layer in layer_list:
#     acc_full_degree.append(data1['results'][layer]['metadata']['accuracy'] * 100)
#     acc_last_token.append(data2['results'][layer]['metadata']['accuracy'] * 100)

# # Create plot
# fig, ax = plt.subplots(figsize=(10, 6))

# ax.plot(selected_layers, acc_full_degree, marker='o', linewidth=2.5, markersize=8, 
#         label='Full Degree', color='#3498db', alpha=0.8)
# ax.plot(selected_layers, acc_last_token, marker='s', linewidth=2.5, markersize=8, 
#         label='Last Token', color='#e74c3c', alpha=0.8)

# # Customize plot
# ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
# ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
# ax.set_title('Intervention Results: InternVL3-1B on TDIUC Counting Task', 
#              fontsize=13, fontweight='bold')
# ax.set_xticks(selected_layers)
# ax.grid(True, alpha=0.3, linestyle='--')
# ax.legend(loc='best', fontsize=11)

# # Set y-axis limits with some padding
# y_min = min(min(acc_full_degree), min(acc_last_token)) - 2
# y_max = max(max(acc_full_degree), max(acc_last_token)) + 2
# ax.set_ylim(y_min, y_max)

# plt.tight_layout()

# # Save figure
# output_path = "results/3.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"Plot saved to: {output_path}")


# import json
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the two hub neuron analysis results
# color_path = "results/hub_neurons/hub_neurons_InternVL3-1B_clevr_color_top90.json"
# counting_path = "results/hub_neurons/hub_neurons_InternVL3-1B_clevr_counting_top90.json"

# with open(color_path, 'r') as f:
#     color_data = json.load(f)

# with open(counting_path, 'r') as f:
#     counting_data = json.load(f)

# # Get selected layers (should be same for both)
# selected_layers = color_data['metadata']['selected_layers']
# criterion = 'full_degree'
# top_n = 90

# # Compute overlap for each layer
# overlaps = []
# task_specific = []  # Non-overlapping neurons (same for both tasks)

# for layer in selected_layers:
#     layer_str = str(layer)
    
#     # Get top 90 neurons for each task
#     color_counter = color_data[criterion][layer_str]
#     counting_counter = counting_data[criterion][layer_str]
    
#     color_top90 = set([int(n[0]) for n in sorted(color_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]])
#     counting_top90 = set([int(n[0]) for n in sorted(counting_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]])
    
#     # Compute overlap and task-specific
#     overlap = len(color_top90 & counting_top90)
#     non_overlap = top_n - overlap  # Same as len(color_top90 - counting_top90) or len(counting_top90 - color_top90)
    
#     overlaps.append(overlap)
#     task_specific.append(non_overlap)
    
#     print(f"Layer {layer}: Overlap={overlap} ({overlap/top_n*100:.1f}%), Task-specific={non_overlap} ({non_overlap/top_n*100:.1f}%)")

# # Create stacked bar plot
# fig, ax = plt.subplots(figsize=(10, 6))

# x = np.arange(len(selected_layers))
# width = 0.6

# # Two-color stack: Overlap (purple) and Task-specific (blue)
# p1 = ax.bar(x, overlaps, width, label='Overlap', color='#9b59b6', alpha=0.8)
# p2 = ax.bar(x, task_specific, width, bottom=overlaps, label='Task-specific', color='#3498db', alpha=0.8)

# # Add value labels on each segment
# for i, layer in enumerate(selected_layers):
#     # Overlap percentage
#     overlap_pct = (overlaps[i] / top_n) * 100
#     if overlaps[i] > 5:
#         ax.text(i, overlaps[i]/2, f'{overlaps[i]}\n({overlap_pct:.1f}%)', 
#                 ha='center', va='center', fontweight='bold', fontsize=10, color='white')
    
#     # Task-specific percentage
#     task_pct = (task_specific[i] / top_n) * 100
#     if task_specific[i] > 5:
#         ax.text(i, overlaps[i] + task_specific[i]/2, f'{task_specific[i]}\n({task_pct:.1f}%)', 
#                 ha='center', va='center', fontweight='bold', fontsize=10, color='white')

# ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
# ax.set_ylabel('Number of Neurons (out of 90)', fontsize=12, fontweight='bold')
# ax.set_title('Top-90 Hub Neuron Overlap: Color vs Counting Tasks (InternVL3-1B)\n(Full Degree Criterion)', 
#              fontsize=13, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels([f'Layer {l}' for l in selected_layers])
# ax.set_ylim(0, 100)
# ax.legend(loc='upper right', fontsize=11)
# ax.grid(axis='y', alpha=0.3, linestyle='--')

# # Add horizontal line at 90
# ax.axhline(y=90, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# plt.tight_layout()
# output_path = "results/hub_neurons/overlap_color_counting_InternVL3-1B_top90.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"\nPlot saved to: {output_path}")
# plt.close()


from datasets import load_dataset, concatenate_datasets
from PIL import Image
import ast
import re

# Load all MMMU subjects
subjects = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 
    'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 
    'Clinical_Medicine', 'Computer_Science', 'Design', 
    'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 
    'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]

print("Loading MMMU dataset...")
all_datasets = []
for subject in subjects:
    dataset = load_dataset("MMMU/MMMU", subject, split="validation")
    all_datasets.append(dataset)

full_dataset = concatenate_datasets(all_datasets)
print(f"Total samples: {len(full_dataset)}")


def parse_options(options):
    """Format options as A. option1, B. option2, etc."""
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def mmmu_doc_to_visual(doc):
    """
    Extract images based on <image N> tokens in the question.
    This is the OFFICIAL way from lmms-eval.
    """
    question = doc["question"]
    # Find all <image N> tokens in the question
    image_tokens = re.findall(r"<image \d+>", question)
    # Remove <> and swap space as underscore: "<image 1>" -> "image_1"
    image_tokens = sorted(list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens])))
    # Get the actual images from doc using the keys
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


def mmmu_doc_to_text(doc, mc_prompt="Answer with the option letter.", open_ended_prompt="Answer the question."):
    """
    Format the question text with options.
    Note: By default, this KEEPS <image N> tokens in the text.
    """
    question = doc["question"]
    
    if doc["question_type"] == "multiple-choice":
        # IMPORTANT: options is stored as STRING in MMMU dataset, need to parse it
        parsed_options = parse_options(ast.literal_eval(doc["options"]))
        question = f"{question}\n{parsed_options}\n\n{mc_prompt}"
    else:
        question = f"{question}\n\n{open_ended_prompt}"
    
    return question


def process_mmmu_sample(doc, interleaved=True):
    """
    Process MMMU sample following the official lmms-eval approach.
    
    Args:
        doc: Sample from MMMU dataset
        interleaved: If True, replace <image N> with <image> for interleaved format
                    If False, keep <image N> tokens as-is
    
    Returns:
        dict with 'images' (list of PIL Images) and 'prompt' (str)
    """
    # Get images - this follows the ORDER of <image N> tokens in question
    images = mmmu_doc_to_visual(doc)
    
    # Get question text
    prompt = mmmu_doc_to_text(doc)
    
    # For interleaved format: replace ALL <image N> with <image>
    # if interleaved:
    #     prompt = re.sub(r"<image \d+>", "<image>", prompt)
    
    return {
        "images": images,
        "prompt": prompt,
        "question_type": doc["question_type"],
        "answer": doc["answer"],
        "id": doc.get("id", ""),
        "options": ast.literal_eval(doc["options"]) if doc["question_type"] == "multiple-choice" else None
    }


# Example usage
print("\n" + "="*80)
print("SAMPLE EXAMPLES:")
print("="*80)

for idx in range(min(3, len(full_dataset))):
    sample = full_dataset[idx]
    
    # For interleaved models (most VLMs)
    processed = process_mmmu_sample(sample, interleaved=True)
    
    print(f"\n--- Sample {idx} ---")
    print(f"ID: {processed['id']}")
    print(f"Question Type: {processed['question_type']}")
    print(f"Number of images: {len(processed['images'])}")
    
    print(f"\nOriginal question (with <image N> tokens):")
    print(f"{sample['question'][:200]}...")
    
    print(f"\nProcessed prompt (interleaved with <image> tokens):")
    print(f"{processed['prompt'][:300]}...")
    
    print(f"\nCorrect Answer: {processed['answer']}")
    print("-" * 80)


# Example inference function
def inference_example(doc):
    """
    Example showing how to use with VLM inference
    """
    # Process the sample
    processed = process_mmmu_sample(doc, interleaved=True)
    
    images = processed['images']  # List of PIL Images in order
    prompt = processed['prompt']   # Text with <image> placeholders
    
    # For models that need interleaved input:
    # Split by <image> and interleave with images
    text_parts = prompt.split("<image>")
    
    # Build interleaved messages (example format)
    messages = []
    for i, (img, text) in enumerate(zip(images, text_parts)):
        if text.strip():
            messages.append({"type": "text", "content": text.strip()})
        messages.append({"type": "image", "content": img})
    # Add remaining text after last image
    if text_parts[-1].strip():
        messages.append({"type": "text", "content": text_parts[-1].strip()})
    
    # Your VLM inference here
    # response = your_vlm_model(messages)
    
    return messages

# Test the inference example
test_sample = full_dataset[0]
messages = inference_example(test_sample)
print("\n" + "="*80)
print("INTERLEAVED MESSAGE FORMAT:")
print("="*80)
for i, msg in enumerate(messages):
    if msg['type'] == 'text':
        print(f"[{i}] TEXT: {msg['content'][:1000]}...")
    else:
        print(f"[{i}] IMAGE: {type(msg['content'])}")

