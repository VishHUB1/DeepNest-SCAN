import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns
from openTSNE import TSNE
import logging
from pathlib import Path
import warnings
from sklearn.metrics import confusion_matrix
import pandas as pd

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = r"C:\Users\visha\Desktop\crack_project\SDNET2018"  # Update with the actual path to SDNET2018 dataset
RESULTS_DIR = "C:/Users/visha/Desktop/crack_project/results_sdnet2018"
IMG_SIZE = (224, 224)
N_CLUSTERS = [2, 4, 6]
PCA_N_COMPONENTS = 50
UMAP_N_COMPONENTS = 10
SOM_GRID = (20, 20)
BATCH_SIZE = 32

# Ensure results directory exists
Path(RESULTS_DIR).mkdir(exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load pre-trained ResNet50
resnet = models.resnet50(pretrained=True).eval().to(device)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_images(base_dir):
    """Load all images from SDNET2018 subdirectories (D, P, W)."""
    categories = {'D': {'cracked': 'CD', 'non_cracked': 'UD'},
                  'P': {'cracked': 'CP', 'non_cracked': 'UP'},
                  'W': {'cracked': 'CW', 'non_cracked': 'UW'}}
    all_data = {}
    for cat, subdirs in categories.items():
        cat_path = os.path.join(base_dir, cat)
        all_data[cat] = {'images': [], 'labels': []}
        for label, subdir in subdirs.items():
            subdir_path = os.path.join(cat_path, subdir)
            if os.path.exists(subdir_path):
                images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(('.jpg', '.png'))]
                all_data[cat]['images'].extend(images)
                all_data[cat]['labels'].extend([1 if label == 'cracked' else 0] * len(images))
    return all_data


def extract_features(image_paths, batch_size=32):
    """Extract features using ResNet50 in batches."""
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = [transform(Image.open(p).convert('RGB')).unsqueeze(0).to(device) for p in batch_paths]
        batch_tensor = torch.cat(batch_imgs, dim=0)
        with torch.no_grad():
            batch_features = resnet(batch_tensor).cpu().numpy()
        features.extend(batch_features)
    return np.array(features)


def reduce_dimensions(features, method='pca'):
    """Reduce dimensionality using PCA or UMAP."""
    if method == 'pca':
        reducer = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
    else:  # umap
        reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, random_state=42)
    return reducer.fit_transform(features)


def evaluate_clusters(features, clusters, labels, method_name, results_dir):
    """Compute clustering metrics and save to file."""
    metrics = {}
    try:
        metrics['silhouette'] = silhouette_score(features, clusters)
        metrics['davies_bouldin'] = davies_bouldin_score(features, clusters)
        metrics['adjusted_rand'] = adjusted_rand_score(labels, clusters)
        metrics['purity'] = purity_score(labels, clusters)
    except Exception as e:
        logger.warning(f"Error computing metrics for {method_name}: {e}")

    # Save metrics
    with open(os.path.join(results_dir, f"{method_name}_metrics.txt"), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

    # Save confusion matrix
    cm = confusion_matrix(labels, clusters)
    pd.DataFrame(cm).to_csv(os.path.join(results_dir, f"{method_name}_confusion.csv"))

    return metrics


def purity_score(y_true, y_pred):
    """Calculate purity score."""
    contingency_matrix = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def plot_tsne(features, clusters, labels, method_name, results_dir):
    """Generate t-SNE visualization."""
    tsne = TSNE(
        n_components=2,
        random_state=42,
        initialization="pca"
    )
    embedding = tsne.fit(features)
    tsne_features = embedding.transform(features)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=clusters, style=labels, palette='deep')
    plt.title(f"t-SNE Visualization - {method_name}")
    plt.savefig(os.path.join(results_dir, f"{method_name}_tsne.png"))
    plt.close()


def plot_pca(features, clusters, labels, method_name, results_dir):
    """Generate PCA visualization."""
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(features)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=clusters, style=labels, palette='deep')
    plt.title(f"PCA Visualization - {method_name}")
    plt.savefig(os.path.join(results_dir, f"{method_name}_pca.png"))
    plt.close()


def plot_cluster_histogram(clusters, method_name, results_dir):
    """Generate histogram of cluster sizes."""
    plt.figure(figsize=(10, 6))
    plt.hist(clusters, bins=max(clusters) + 1, edgecolor='black')
    plt.title(f"Cluster Size Distribution - {method_name}")
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Images")
    plt.savefig(os.path.join(results_dir, f"{method_name}_histogram.png"))
    plt.close()


def plot_som(som, features, labels, method_name, results_dir):
    """Generate SOM grid visualization."""
    plt.figure(figsize=(10, 8))
    winner_coordinates = np.array([som.winner(x) for x in features]).T
    plt.scatter(winner_coordinates[0], winner_coordinates[1], c=labels, cmap='viridis')
    plt.title(f"SOM Grid - {method_name}")
    plt.savefig(os.path.join(results_dir, f"{method_name}_som.png"))
    plt.close()


def main():
    # Load images for each category (D, P, W)
    all_data = load_images(BASE_DIR)

    for category in ['D', 'P', 'W']:
        logger.info(f"Processing category: {category}")
        image_paths = all_data[category]['images']
        true_labels = all_data[category]['labels']
        logger.info(f"Loaded {len(image_paths)} images for {category}")

        # Create category-specific results directory
        cat_results_dir = os.path.join(RESULTS_DIR, category)
        Path(cat_results_dir).mkdir(exist_ok=True)

        logger.info(f"Extracting features for {category}...")
        features = extract_features(image_paths, batch_size=BATCH_SIZE)
        logger.info(f"Extracted features shape for {category}: {features.shape}")

        # Save features to disk to free memory
        np.save(os.path.join(cat_results_dir, "features.npy"), features)
        torch.cuda.empty_cache()

        # Dimensionality reduction
        logger.info(f"Reducing dimensions with PCA for {category}...")
        pca_features = reduce_dimensions(features, method='pca')
        logger.info(f"Reducing dimensions with UMAP for {category}...")
        umap_features = reduce_dimensions(features, method='umap')

        # Clustering methods
        clustering_methods = [
            ('kmeans', lambda x, k: MiniBatchKMeans(n_clusters=k, random_state=42).fit_predict(x)),
            ('birch', lambda x, k: Birch(n_clusters=k, threshold=0.5).fit_predict(x))
        ]

        for method_name, cluster_func in clustering_methods:
            logger.info(f"Running {method_name} clustering for {category}...")
            for k in N_CLUSTERS:
                clusters_pca = cluster_func(pca_features, k)
                clusters_umap = cluster_func(umap_features, k)

                # Evaluate and save results
                metrics_pca = evaluate_clusters(pca_features, clusters_pca, true_labels, f"{method_name}_k{k}_pca",
                                                cat_results_dir)
                metrics_umap = evaluate_clusters(umap_features, clusters_umap, true_labels, f"{method_name}_k{k}_umap",
                                                 cat_results_dir)
                logger.info(f"{method_name} (k={k}) PCA metrics for {category}: {metrics_pca}")
                logger.info(f"{method_name} (k={k}) UMAP metrics for {category}: {metrics_umap}")

                # Visualize
                plot_tsne(pca_features, clusters_pca, true_labels, f"{method_name}_k{k}_pca", cat_results_dir)
                plot_tsne(umap_features, clusters_umap, true_labels, f"{method_name}_k{k}_umap", cat_results_dir)
                plot_pca(pca_features, clusters_pca, true_labels, f"{method_name}_k{k}_pca", cat_results_dir)
                plot_pca(umap_features, clusters_umap, true_labels, f"{method_name}_k{k}_umap", cat_results_dir)
                plot_cluster_histogram(clusters_pca, f"{method_name}_k{k}_pca", cat_results_dir)
                plot_cluster_histogram(clusters_umap, f"{method_name}_k{k}_umap", cat_results_dir)

        # SOM clustering
        logger.info(f"Running SOM clustering for {category}...")
        som = MiniSom(SOM_GRID[0], SOM_GRID[1], features.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
        som.train(features, 5000)
        clusters_som = [som.winner(x)[0] * SOM_GRID[1] + som.winner(x)[1] for x in features]
        metrics_som = evaluate_clusters(features, clusters_som, true_labels, "som", cat_results_dir)
        logger.info(f"SOM metrics for {category}: {metrics_som}")
        plot_som(som, features, true_labels, "som", cat_results_dir)
        plot_cluster_histogram(clusters_som, "som", cat_results_dir)


if __name__ == "__main__":
    main()