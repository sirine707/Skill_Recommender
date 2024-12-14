import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def analyze_correlations(df):
    """Analyze feature correlations"""
    corr_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    return corr_matrix

def select_features_importance(X, y, feature_names):
    """Select features using Random Forest importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance

def select_statistical_features(X, y, feature_names, k=10):
    """Select features using statistical tests"""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    
    # Get selected feature mask
    selected_features_mask = selector.get_support()
    selected_features = feature_names[selected_features_mask]
    
    return selected_features

def descriptive_analysis(df):
    """
    Perform descriptive analysis on the dataset.
    """
    # Numerical features
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    print("Descriptive Statistics for Numerical Features:")
    print(df[numerical_cols].describe())
    
    # Calculate additional statistics
    for col in numerical_cols:
        print(f"\nStatistics for {col}:")
        print(f"Mean: {df[col].mean()}")
        print(f"Median: {df[col].median()}")
        print(f"Variance: {df[col].var()}")
        print(f"Standard Deviation: {df[col].std()}")
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("\nDistribution of Categorical Features:")
    for col in categorical_cols:
        print(f"\nDistribution for {col}:")
        print(df[col].value_counts())

    # Visualizations
    for col in numerical_cols:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram of {col}")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()

    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Bar Chart of {col}")
        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()

def correlation_analysis(df):
    """
    Analyze correlations between variables in the dataset.
    """
    # Numerical features correlation
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Pearson correlation
    pearson_corr = df[numerical_cols].corr(method='pearson')
    print("Pearson Correlation Coefficients:")
    print(pearson_corr)
    
    # Spearman correlation
    spearman_corr = df[numerical_cols].corr(method='spearman')
    print("\nSpearman Correlation Coefficients:")
    print(spearman_corr)
    
    # Heatmap for Pearson correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Pearson Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('pearson_correlation_heatmap.png')
    plt.close()
    
    # Heatmap for Spearman correlation
    plt.figure(figsize=(10, 8))
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Spearman Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('spearman_correlation_heatmap.png')
    plt.close()
    
    # Chi-square test for categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("\nChi-square Test Results:")
    for col in categorical_cols:
        for col2 in categorical_cols:
            if col != col2:
                contingency_table = pd.crosstab(df[col], df[col2])
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                print(f"Chi-square test between {col} and {col2}: p-value = {p}")
    
    # Point-biserial correlation for binary vs numerical
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    
    print("\nPoint-biserial Correlation Results:")
    for bin_col in binary_cols:
        for num_col in numerical_cols:
            corr, p_value = pointbiserialr(df[bin_col], df[num_col])
            print(f"Point-biserial correlation between {bin_col} and {num_col}: correlation = {corr}, p-value = {p_value}")

def clustering_analysis(df_scaled):  # Use already scaled data
    """
    Explore clustering possibilities for students.
    """
    # Use df_scaled directly
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(df_scaled)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(df_scaled)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=3)
    hierarchical_labels = hierarchical.fit_predict(df_scaled)
    
    # Evaluate Clustering
    print("\nClustering Evaluation:")
    for labels, method in zip([kmeans_labels, dbscan_labels, hierarchical_labels],
                              ['K-Means', 'DBSCAN', 'Hierarchical']):
        if len(set(labels)) > 1:  # Check if more than one cluster is formed
            silhouette_avg = silhouette_score(df_scaled, labels)
            davies_bouldin = davies_bouldin_score(df_scaled, labels)
            print(f"{method} - Silhouette Score: {silhouette_avg:.2f}, Davies-Bouldin Index: {davies_bouldin:.2f}")
        else:
            print(f"{method} - Only one cluster formed.")
    
    # Visualize Clusters using PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)
    
    plt.figure(figsize=(12, 4))
    for i, (labels, method) in enumerate(zip([kmeans_labels, dbscan_labels, hierarchical_labels],
                                             ['K-Means', 'DBSCAN', 'Hierarchical']), 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=labels, palette='viridis')
        plt.title(f'{method} Clustering')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.savefig('clustering_visualization.png')
    plt.close()


def main(df):
    # 1. Preprocess numerical features
    df_num_scaled, scaler = preprocess_numerical_features(df)
    
    # 2. Preprocess categorical features
    df_cat_encoded, encoders = preprocess_categorical_features(df)
    
    # 3. Combine preprocessed features
    df_processed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    
    # 4. Analyze correlations
    corr_matrix = analyze_correlations(df_processed)
    
    # 5. Prepare data for feature selection
    X = df_processed.drop('score_final', axis=1)
    y = df_processed['score_final']
    
    # 6. Get feature importance
    feature_importance = select_features_importance(X, y, X.columns)
    
    # 7. Get statistical feature selection
    selected_features = select_statistical_features(X, y, X.columns)
    
    # 8. Perform descriptive analysis
    descriptive_analysis(df)
    
    # 9. Perform correlation analysis
    correlation_analysis(df)
    
    # 10. Perform clustering analysis
    clustering_analysis(df_num_scaled)  # Pass pre-scaled data
    
    return {
        'processed_data': df_processed,
        'feature_importance': feature_importance,
        'selected_features': selected_features,
        'correlation_matrix': corr_matrix,
        'scaler': scaler,
        'encoders': encoders
    }

if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('Base_Etudiant.csv')  # Adjust path as needed
    
    result = main(df)
    
    # Print feature selection summary
    print("\nFeature Selection Summary:")
    print("==========================")
    print("\nTop Features by Random Forest:")
    for i, feature in enumerate(result['feature_importance']['feature'].head(10), 1):
        print(f"{i}. {feature}")
    
    print("\nStatistically Selected Features:")
    for i, feature in enumerate(result['selected_features'], 1):
        print(f"{i}. {feature}")
