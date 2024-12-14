from .preprocess import preprocess_text
from .vectorize import vectorize_text
from .visualizations import visualize_ngrams
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import dcor
from scipy.stats import pearsonr, spearmanr, kendalltau

def extract_ngrams(text, n=2):
    """Extract n-grams from text."""
    tokens = text.split()
    ngrams = zip(*[tokens[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]
# Function to perform PCA or t-SNE based on input flag
def reduce_dimensionality(data, method='PCA', n_components=2):
    if method == 'PCA':
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
    elif method == 't-SNE':
        tsne = TSNE(n_components=n_components)
        reduced_data = tsne.fit_transform(data)
    else:
        raise ValueError("Invalid method. Choose 'PCA' or 't-SNE'.")
    return reduced_data

# Function to compute distance covariance with pairwise distance
def compute_dcov_with_cdist(X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    distance_matrix = cdist(X, Y, metric='euclidean')
    return np.mean(distance_matrix)

# Function to compute correlation coefficients
def compute_correlation_coefficients(X, Y):
    pearson_corr, _ = pearsonr(X.flatten(), Y.flatten())
    spearman_corr, _ = spearmanr(X.flatten(), Y.flatten())
    kendall_corr, _ = kendalltau(X.flatten(), Y.flatten())
    return pearson_corr, spearman_corr, kendall_corr

# Function to compute RV Coefficient
def compute_rv_coefficient(X, Y):
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    rv_coefficient = np.dot(X_flat, Y_flat) / (np.linalg.norm(X_flat) * np.linalg.norm(Y_flat))
    return rv_coefficient

# Process and align data
def process_file(file_path):
    if file_path.endswith('.csv'):
        data_df = pd.read_csv(file_path).fillna('').astype(str)
    elif file_path.endswith('.json'):
        data_df = pd.read_json(file_path).fillna('').astype(str)
    
    for col in data_df.columns:
        data_df[col] = data_df[col].apply(preprocess_text)

    return data_df

# Function to align columns across all dataframes
def align_columns(data_frames):
    all_columns = set()
    for df in data_frames:
        all_columns.update(df.columns)
    
    aligned_data_frames = []
    for df in data_frames:
        for col in all_columns:
            if col not in df.columns:
                df[col] = ''
        aligned_data_frames.append(df[sorted(all_columns)])
    
    return aligned_data_frames

# Vectorize the data
def vectorize_data(data_frames):
    vectors_list = []
    for df in data_frames:
        vectors = {}
        for col in df.columns:
            vectors[col] = np.array([vectorize_text(str(text)) for text in df[col]])
        matrices = [np.array(vectors[col]) for col in df.columns]
        combined_vectors = np.hstack(matrices)
        vectors_list.append(combined_vectors)
    return vectors_list

# Main pipeline to run the process
def run_pipeline(file_paths, output_path, dim_reduction_method='PCA', n_components=2):
    # Process each file and store the results
    data_frames = [process_file(file_path) for file_path in file_paths]

    # Align columns across all dataframes
    aligned_data_frames = align_columns(data_frames)
    visualize_ngrams(aligned_data_frames, n=2, top_n=20)

    # Vectorize the data
    combined_vectors = vectorize_data(aligned_data_frames)
    all_combined_vectors = np.vstack(combined_vectors)
    
    # Perform dimensionality reduction (PCA or t-SNE)
    reduced_vectors = reduce_dimensionality(all_combined_vectors, method=dim_reduction_method, n_components=n_components)

    # Save the reduced vectors to a CSV file
    reduced_df = pd.DataFrame(reduced_vectors)
    reduced_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Reduced vectors saved to '{output_path}'")

    # Compute similarities between the files
    similarities = {}
    for i in range(len(combined_vectors)):
        for j in range(i + 1, len(combined_vectors)):
            # Ensure 2D input for cdist
            dcov_similarity = compute_dcov_with_cdist(reduced_vectors[i], reduced_vectors[j])
            similarities[f"File {i+1} vs File {j+1}"] = {'distance_covariance': dcov_similarity}

            # Compute RV coefficient
            rv_similarity = compute_rv_coefficient(reduced_vectors[i], reduced_vectors[j])
            similarities[f"File {i+1} vs File {j+1}"].update({'rv_coefficient': rv_similarity})

            # Compute Correlation Coefficients
            pearson_corr, spearman_corr, kendall_corr = compute_correlation_coefficients(reduced_vectors[i], reduced_vectors[j])
            similarities[f"File {i+1} vs File {j+1}"].update({
                'pearson_correlation': pearson_corr,
                'spearman_correlation': spearman_corr,
                'kendall_correlation': kendall_corr
            })

            print(f"Distance Covariance: {dcov_similarity}, RV Coefficient: {rv_similarity}")
            print(f"Pearson Correlation: {pearson_corr}, Spearman Correlation: {spearman_corr}, Kendall Correlation: {kendall_corr}")

    return similarities
