# import matplotlib.pyplot as plt
# import numpy as np

# # Raw data
# scales = np.array([0, 0.1, 0.5, 2.0, -0.5, -1, -0.1])
# acc_5 = np.array([99.6, 99.6, 99.8, 99.8, 99.7, 99.3, 99.7])
# acc_10 = np.array([13.9, 16, 99, 99.5, 12.6, 13.7, 13.5])
# baseline_acc = 99.8

# # ---- Sort by scaling factor ----
# sorted_idx = np.argsort(scales)
# scales = scales[sorted_idx]
# acc_5 = acc_5[sorted_idx]
# acc_10 = acc_10[sorted_idx]

# # ---- Plot ----
# plt.figure(figsize=(8,5))

# plt.plot(scales, acc_5, marker='o', label="Intervene 5 neurons")
# plt.plot(scales, acc_10, marker='o', label="Intervene 10 neurons")

# plt.axhline(y=baseline_acc, linestyle="--", label="No intervention baseline (99.8%)")

# plt.title("Qwen – Neuron Intervention Effect on Accuracy")
# plt.xlabel("Scaling factor applied to selected neurons")
# plt.ylabel("Accuracy (%)")
# plt.xticks(scales)
# plt.legend()
# plt.tight_layout()


# plt.savefig("results/neurons_intervention_effect_qwen.png", dpi=300)


# import matplotlib.pyplot as plt

# # Data
# methods = ["Topology", "Activation", "Random"]
# accuracy = [14, 99.7, 99.8]  # corresponding accuracies

# # Plot
# plt.figure(figsize=(6,4))
# plt.bar(methods, accuracy)

# plt.title("Qwen – Ablation: Removing Top 1% Neurons (Layer 0)")
# plt.ylabel("Accuracy (%)")
# plt.tight_layout()
# plt.savefig("results/ablation_top1_percent_qwen.png", dpi=300)



import numpy as np

data = np.load("results/modality_correlation/modality_corr_Qwen2.5-VL-3B_tdiuc_counting.npy", allow_pickle=True).item()

print(data['vv_mean'])