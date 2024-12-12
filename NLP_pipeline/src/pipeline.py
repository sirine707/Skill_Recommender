import pandas as pd
import numpy as np
from .preprocess import preprocess_text
from .vectorize import vectorize_text

def process_file(file_path):
    # Load the CSV data
    data_df = pd.read_csv(file_path).fillna('').astype(str)

    # Preprocess each column
    for col in data_df.columns:
        data_df[col] = data_df[col].apply(preprocess_text)

    # Vectorize each column
    columns_to_vectorize = data_df.columns
    vectors = {}
    for col in columns_to_vectorize:
        vectors[col] = np.array([vectorize_text(str(text)) for text in data_df[col]])

    # Combine the vectors horizontally
    matrices = [np.array(vectors[col]) for col in columns_to_vectorize]
    combined_vectors = np.hstack(matrices)

    return combined_vectors

def run_pipeline(file_paths, output_path):
    # Process each file and store the results
    processed_data = [process_file(file_path) for file_path in file_paths]

    # Combine all processed data into a single array
    all_combined_vectors = np.vstack(processed_data)

    # Save the combined vectors to a CSV file
    combined_df = pd.DataFrame(all_combined_vectors)
    combined_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"All files have been processed and combined vectors saved to '{output_path}'")