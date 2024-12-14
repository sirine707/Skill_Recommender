from collections import Counter
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
def extract_ngrams(text, n=2):
    """Extract n-grams from text."""
    vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([text])
    ngram_features = vectorizer.get_feature_names_out()
    return ngram_features

def identify_text_column(df):
    """Identify the most likely text column in a data frame."""
    for column in df.columns:
        if df[column].dtype == 'object' and df[column].str.contains(r'\w').any():
            return column
    raise ValueError("No suitable text column found")

def visualize_ngrams(data_frames, n=2, top_n=20):
    """Visualize the top n n-grams."""
    all_ngrams = []
    for df in data_frames:
        text_column = identify_text_column(df)
        for text in df[text_column]:
            all_ngrams.extend(extract_ngrams(text, n=n))
    
    ngram_counts = Counter(all_ngrams)
    ngram_df = pd.DataFrame(ngram_counts.items(), columns=['ngram', 'count']).sort_values(by='count', ascending=False).head(top_n)
    
    # Visualize using Plotly
    fig = px.bar(ngram_df, x='ngram', y='count', title=f'Top {top_n} {n}-grams', template='plotly_white', labels={'ngram': 'N-gram', 'count': 'Count'})
    fig.show()

def visualize_bigrams(data_frames, top_n=20):
    """Visualize the top n bigrams."""
    all_bigrams = []
    for df in data_frames:
        text_column = identify_text_column(df)
        for text in df[text_column]:
            all_bigrams.extend(extract_ngrams(text, n=2))
    
    bigram_counts = Counter(all_bigrams)
    bigram_df = pd.DataFrame(bigram_counts.items(), columns=['ngram', 'count']).sort_values(by='count', ascending=False).head(top_n)
    
    # Visualize using Plotly
    fig = px.bar(bigram_df[:20], x='ngram', y='count', title='Counts of top bigrams', template='plotly_white', labels={'ngram': 'Bigram', 'count': 'Count'})
    fig.show()

def compare_ngrams_between_files(data_frames, n=2, top_n=20):
    """Compare the top n n-grams between files."""
    ngrams_list = []
    for df in data_frames:
        text_column = identify_text_column(df)
        all_ngrams = []
        for text in df[text_column]:
            all_ngrams.extend(extract_ngrams(text, n=n))
        ngram_counts = Counter(all_ngrams)
        ngram_df = pd.DataFrame(ngram_counts.items(), columns=['ngram', 'count']).sort_values(by='count', ascending=False).head(top_n)
        ngrams_list.append(ngram_df)

    # Combine the data frames for comparison
    combined_df = pd.concat(ngrams_list, axis=1)
    combined_df.columns = ['ngram', 'count1', 'ngram', 'count2', 'ngram', 'count3']  # Adjust based on the number of files
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Melt the data frame for Plotly
    long_bigram_df_tidy = pd.melt(combined_df, id_vars=['ngram'], value_vars=['count1', 'count2', 'count3'], var_name='variable', value_name='value')

    # Visualize using Plotly
    fig = px.bar(long_bigram_df_tidy, title='Comparison of Top N-Grams Between Files', x='ngram', y='value',
                 color='variable', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Bold,
                 labels={'variable': 'File:', 'ngram': 'N-Gram'})
    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=0.1, y=1.1))
    fig.update_yaxes(title='', showticklabels=False)
    fig.show()

def compare_words_between_files(data_frames, top_n=20):
    """Compare the most frequent words between files."""
    words_list = []
    for df in data_frames:
        text_column = identify_text_column(df)
        all_words = []
        for text in df[text_column]:
            all_words.extend(text.split())
        word_counts = Counter(all_words)
        word_df = pd.DataFrame(word_counts.items(), columns=['word', 'count']).sort_values(by='count', ascending=False).head(top_n)
        words_list.append(word_df)

    # Combine the data frames for comparison
    combined_df = pd.concat(words_list, axis=1)
    combined_df.columns = ['word', 'count1', 'word', 'count2', 'word', 'count3']  # Adjust based on the number of files
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    # Melt the data frame for Plotly
    long_word_df_tidy = pd.melt(combined_df, id_vars=['word'], value_vars=['count1', 'count2', 'count3'], var_name='variable', value_name='value')

    # Visualize using Plotly
    fig = px.bar(long_word_df_tidy, title='Comparison of Most Frequent Words Between Files', x='word', y='value',
                 color='variable', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Bold,
                 labels={'variable': 'File:', 'word': 'Word'})
    fig.update_layout(legend_orientation="h")
    fig.update_layout(legend=dict(x=0.1, y=1.1))
    fig.update_yaxes(title='', showticklabels=False)
    fig.show()

def visualize_bigram_similarity(data_frames, top_n=20, min_freq=2):
    # Collect bigrams
    all_bigrams = []
    for df in data_frames:
        text_column = identify_text_column(df)
        for text in df[text_column]:
            if isinstance(text, str) and text.strip():
                bigrams = extract_ngrams(text, n=2)
                all_bigrams.extend(bigrams)
    
    # Count and filter bigrams
    bigram_counts = Counter(all_bigrams)
    bigram_df = pd.DataFrame(bigram_counts.items(), columns=['bigram', 'count'])
    bigram_df = bigram_df.sort_values(by='count', ascending=False).head(top_n)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    bigram_texts = [' '.join(b) if isinstance(b, tuple) else str(b) for b in bigram_df['bigram']]
    bigram_matrix = vectorizer.fit_transform(bigram_texts)
    
    # Compute t-SNE with error handling
    n_samples = bigram_matrix.shape[0]
    perplexity = min(30, max(5, n_samples - 1))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_embeddings = tsne.fit_transform(bigram_matrix.toarray())
    
    # Create visualization DataFrame with explicit size calculation
    embed_df = pd.DataFrame(tsne_embeddings, columns=['tsne_1', 'tsne_2'])
    embed_df['bigram'] = bigram_texts
    embed_df['count'] = bigram_df['count'].values
    
    # Normalize counts for size, handling edge cases
    min_count = embed_df['count'].min()
    max_count = embed_df['count'].max()
    
    if min_count == max_count:
        embed_df['size'] = 20  # Use constant size if all counts are equal
    else:
        # Scale sizes between 10 and 50
        embed_df['size'] = 10 + ((embed_df['count'] - min_count) / (max_count - min_count)) * 40
    
    # Verify no NaN values
    embed_df = embed_df.dropna(subset=['size', 'count'])
    
    if embed_df.empty:
        print("No valid data points after cleaning")
        return
    
    # Create scatter plot
    fig = px.scatter(
        embed_df,
        x='tsne_1',
        y='tsne_2',
        hover_name='bigram',
        text='bigram',
        size='size',
        color='count',
        size_max=45,
        template='plotly_white',
        title='Bigram Similarity and Frequency'
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='Gray')))
    fig.show()


def visualize_topic_distributions(data_frames, n_topics=5, top_n=10, summary_words=3):
    """Visualize topic distributions using LDA and Plotly."""
    # Combine all text data
    all_texts = []
    for df in data_frames:
        text_column = identify_text_column(df)
        all_texts.extend(df[text_column])

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    text_matrix = vectorizer.fit_transform(all_texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_matrix)

    # Get topic distributions
    topic_distributions = lda.transform(text_matrix)
    topic_words = vectorizer.get_feature_names_out()

    # Get the top words for each topic
    top_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words.append([topic_words[i] for i in topic.argsort()[:-top_n - 1:-1]])

    # Prepare data for visualization
    topic_df = pd.DataFrame(topic_distributions, columns=[f'Topic {i+1}' for i in range(n_topics)])
    topic_df['text'] = all_texts

    # Create topic labels with summarized top words
    topic_labels = [f"Topic {i+1}: " + ", ".join(words[:summary_words]) for i, words in enumerate(top_words)]

    # Visualize topic distributions using a heatmap
    fig = px.imshow(topic_df.drop(columns=['text']).T, labels=dict(x="Document", y="Topic", color="Distribution"),
                    x=topic_df.index, y=topic_labels, aspect="auto", title="Topic Distributions")
    fig.update_layout(coloraxis_showscale=True)
    fig.show()

    # Print the topics and their top words
    print("\nTopics and their top words:")
    for i, words in enumerate(top_words):
        print(f"Topic {i+1}: {', '.join(words[:summary_words])}")

    """Visualize topic distributions using LDA and Plotly."""
    # Combine all text data
    all_texts = []
    for df in data_frames:
        text_column = identify_text_column(df)
        all_texts.extend(df[text_column])

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words='english')
    text_matrix = vectorizer.fit_transform(all_texts)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(text_matrix)

    # Get topic distributions
    topic_distributions = lda.transform(text_matrix)
    topic_words = vectorizer.get_feature_names_out()

    # Get the top words for each topic
    top_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words.append([topic_words[i] for i in topic.argsort()[:-top_n - 1:-1]])

    # Prepare data for visualization
    topic_df = pd.DataFrame(topic_distributions, columns=[f'Topic {i+1}' for i in range(n_topics)])
    topic_df['text'] = all_texts

    # Create a DataFrame for the top words
    top_words_df = pd.DataFrame(top_words, index=[f'Topic {i+1}' for i in range(n_topics)], columns=[f'Word {i+1}' for i in range(top_n)])

    # Visualize topic distributions using a heatmap
    fig = px.imshow(topic_df.drop(columns=['text']).T, labels=dict(x="Document", y="Topic", color="Distribution"),
                    x=topic_df.index, y=topic_df.columns[:-1], aspect="auto", title="Topic Distributions")
    fig.update_layout(coloraxis_showscale=True)
    fig.show()

    # Visualize the top words for each topic
    fig_top_words = px.imshow(top_words_df.T, text_auto=True, aspect='auto', title='Top Words for Each Topic',
                              labels={'x': 'Topic', 'y': 'Top Words'}, template='plotly_white')
    fig_top_words.update_layout(coloraxis_showscale=False)
    fig_top_words.show()