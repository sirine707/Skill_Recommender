from src.pipeline import run_pipeline

# List of file paths
file_paths = ['data/licence.csv', 'data/masters.csv', 'scrapped_content_ref/file3.csv']
output_path = 'output/combined_vectors.csv'

# Run the pipeline
run_pipeline(file_paths, output_path)