from src.pipeline import run_pipeline
from src.pipeline import run_pipeline
from src.visualizations import visualize_ngrams, visualize_bigrams
from src.pipeline import process_file
import pandas as pd
from src.visualizations import compare_ngrams_between_files, visualize_bigram_similarity, visualize_topic_distributions
from tabulate import tabulate #pip install tabulate
from termcolor import colored

from src.analysis import analyze_skills_gap

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
# Load the processed data frames
data_frames = [process_file(file_path) for file_path in file_paths]

# Visualize the top n-grams

def print_recommendations_table(recommendations):
    """Print recommendations in formatted tables."""
    for program, rec in recommendations.items():
        # Header
        print(f"\n{colored('Program Analysis: ' + program, 'blue', attrs=['bold'])}")

        # Summary Table
        summary_data = [
            ['Priority Level', colored(rec['summary']['priority_level'], get_priority_color(rec['summary']['priority_level']))],
            ['Total Skill Gaps', rec['summary']['total_gaps']]
        ]
        print("\nSummary:")
        print(tabulate(summary_data, tablefmt='grid'))

        # Priority Areas Table
        priority_data = [[area, f"{score:.2f}%"] for area, score in rec['priorities'].items()]
        print("\nPriority Areas:")
        print(tabulate(
            priority_data,
            headers=['Category', 'Priority Score'],
            tablefmt='grid'
        ))

        # Actions Table
        action_data = [[i, action] for i, action in enumerate(rec['actions'], 1)]
        print("\nRecommended Actions:")
        print(tabulate(
            action_data,
            headers=['#', 'Action Item'],
            tablefmt='grid'
        ))
        print("\n" + "="*80)

def get_priority_color(priority):
    """Get color based on priority level."""
    colors = {
        'Critical': 'red',
        'High': 'yellow',
        'Medium': 'cyan',
        'Low': 'green'
    }
    return colors.get(priority, 'white')

# Usage in main.py
results = analyze_skills_gap(data_frames)
print_recommendations_table(results['recommendations'])
# Visualize the top n-grams
visualize_ngrams(data_frames, n=2, top_n=20)
# Visualize the top bigrams
visualize_bigrams(data_frames, top_n=20)
# Compare n-grams between files
compare_ngrams_between_files(data_frames, n=2, top_n=20)
# Visualize bigram similarity and frequency using t-SNE embeddings
visualize_bigram_similarity(data_frames, top_n=20,min_freq=2)
# Visualize topic distributions using LDA
visualize_topic_distributions(data_frames, n_topics=5)