import pickle
import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Load features
with open("Data/features.pkl", "rb") as f:
    dataset = pickle.load(f)

X = np.array(dataset['features']) 
y = np.array(dataset['labels'])    

# Optional: color palette
palette = sns.color_palette("bright", len(set(y)))

def plot_embedding(embedding, title, method="pca"):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(set(y)):
        idx = [j for j, val in enumerate(y) if val == label]
        x = [embedding[j][0] for j in idx]
        y_ = [embedding[j][1] for j in idx]
        plt.scatter(x, y_, label=label, alpha=0.6, s=20)
    plt.legend()
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/{method}_projection.png")
    plt.show()

# Create output folder

os.makedirs("visualizations", exist_ok=True)

# PCA Projection
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plot_embedding(X_pca, "PCA Projection of Weather Embeddings", method="pca")

# t-SNE Projection
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, "t-SNE Projection of Weather Embeddings", method="tsne")