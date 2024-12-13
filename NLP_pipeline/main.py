from src.pipeline import run_pipeline

# List of file paths
file_paths = ['data/licence.csv', 'data/masters.csv', 'scrapped_content_ref/file3.csv']
output_path = 'output/combined_vectors.csv'
# Run the pipeline
similarities = run_pipeline(file_paths, output_path,'PCA',2)

# Print the similarities
print("Similarities between files:")
for key, value in similarities.items():
    print(f"{key}:")
    for metric, metric_value in value.items():
        print(f"  {metric}: {metric_value}")