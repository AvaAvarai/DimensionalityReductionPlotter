import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import datetime

# Function to open a file dialog and select CSV file
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

# Function to normalize numerical columns except for the class column
def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Find class column (case-insensitive)
    class_col = [col for col in df.columns if col.lower() == 'class']
    if not class_col:
        raise ValueError("No 'class' column found in the dataset.")
    class_col = class_col[0]

    # Separate features and class labels
    X = df.drop(columns=[class_col])
    y = df[class_col].astype(str)  # Convert to string for categorical handling

    # Normalize numerical features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Function to compute embeddings and visualize them
def plot_embeddings(X, y):
    # Compute embeddings
    pca = PCA(n_components=2).fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, n_jobs=1).fit_transform(X)
    umap_embedding = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(X)

    # Plot settings
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    methods = {'PCA': pca, 't-SNE': tsne, 'UMAP': umap_embedding}

    for ax, (method, embedding) in zip(axes, methods.items()):
        df_vis = pd.DataFrame(embedding, columns=['Dim 1', 'Dim 2'])
        df_vis['class'] = y

        sns.scatterplot(
            x='Dim 1', y='Dim 2', hue='class', data=df_vis, ax=ax, palette='tab10', s=30, alpha=0.7
        )
        ax.set_title(f'{method} Projection')
        ax.legend(loc='best', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    filename = f"embeddings_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(filename)

# Main execution
if __name__ == "__main__":
    file_path = select_file()
    if file_path:
        X, y = preprocess_data(file_path)
        plot_embeddings(X, y)
