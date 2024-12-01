import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

def load_eeg_data(mat_file_path):
    mat_contents = sio.loadmat(mat_file_path)
    eeg_data = mat_contents['data']
    min_val = np.min(eeg_data, axis=1, keepdims=True)
    max_val = np.max(eeg_data, axis=1, keepdims=True)
    return 2 * (eeg_data - min_val) / (max_val - min_val + 1e-8) - 1

def compute_shannon_entropy(signal, num_bins=256):
    histogram_counts, bin_edges = np.histogram(signal, bins=num_bins)
    probabilities = histogram_counts / np.sum(histogram_counts)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def plot_signal(signal, epoch, batch_idx, prefix='Real'):
    signal_np = signal.cpu().numpy().squeeze()
    entropy = compute_shannon_entropy(signal_np)
    plt.figure(figsize=(10, 4))
    plt.plot(signal_np, label='Signal')
    plt.legend()
    plt.title(f"{prefix} Signal - Epoch {epoch}, Batch {batch_idx}\nEntropy: {entropy:.4f}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    os.makedirs("logs/signals", exist_ok=True)
    plt.savefig(f"logs/signals/{prefix}_signal_epoch{epoch}_batch{batch_idx}.png")
    plt.close()

def plot_signal_to_image(signal, epoch, batch_idx, prefix='Real'):
    signal_np = signal.cpu().numpy().squeeze()
    entropy = compute_shannon_entropy(signal_np)
    plt.figure(figsize=(10, 4))
    plt.plot(signal_np, label='Signal')
    plt.legend()
    plt.title(f"{prefix} Signal - Epoch {epoch}, Batch {batch_idx}\nEntropy: {entropy:.4f}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = Image.open(buf)
    return torchvision.transforms.ToTensor()(image)