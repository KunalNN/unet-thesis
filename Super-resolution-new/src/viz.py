# src/utils/viz.py
import matplotlib.pyplot as plt

def plot_sr_triplet(lr, hr, sr, figsize=(12,4)):
    plt.figure(figsize=figsize)
    titles = ["Low-res (input)", "Super-res (pred)", "High-res (ground truth)"]
    for i, img in enumerate([lr, sr, hr]):
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()
