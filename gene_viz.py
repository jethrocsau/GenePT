import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.utils import resample
from scipy.spatial.distance import pdist
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import sys
from sklearn.decomposition import PCA
import gc
import argparse

# ==========================================
# 1. NATURE-STYLE CONFIGURATION
# ==========================================
def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'axes.linewidth': 0.6,
        'figure.dpi': 300,
        'savefig.transparent': True,
        'legend.frameon': False
    })

set_nature_style()

COLORS = {
    'GenePT (Original)': '#333333',
    'GenePT (Proj)': '#E64B35',
    'GenePT Concat (Proj + Original)': '#8491B4', # GenePT Concat
    'GenePT Concat + PCA': '#91D1C2', # PCA reduced GenePT

    'Evo2 (Proj)': '#F39B7F',
    'Evo2 (Mean)': '#00A087',
    'Evo2 Concat (Proj + Mean)': '#4DBBD5', # Evo2 concat
    'Evo2 Concat + PCA': '#7E6148', # PCA reduced Evo2

    'Cross Concat (Evo2 Proj + GenePT Proj)': '#B8860B', # cross concat
    'Original Concat (Evo2 Mean + GenePT Original)': '#8A2BE2', # original concat
}

GENE2VEC_FILE = './data/pre_trained_emb/gene2vec_dim_200_iter_9.txt'

# ==========================================
# 2. CORE UTILITIES
# ==========================================
def build_ggi_features(df, embed_dict):
    X, y = [], []
    valid_genes = set(embed_dict.keys())
    for _, row in df.iterrows():
        g1, g2, label = row['gene_1'], row['gene_2'], row['label']
        if g1 in valid_genes and g2 in valid_genes:
            feat = embed_dict[g1] + embed_dict[g2]
            X.append(feat)
            y.append(label)
    return np.array(X), np.array(y)

def concat_embeddings(embed_a, embed_b):
    common_genes = set(embed_a.keys()) & set(embed_b.keys())
    if not common_genes:
        return {}
    return {g: np.concatenate([embed_a[g], embed_b[g]]) for g in common_genes}

def reduce_dimensions(embed_dict, n_components):
    """
    Reduces the dimensionality of embeddings using PCA.
    """
    genes = list(embed_dict.keys())
    if not genes:
        return {}

    X = np.array([embed_dict[g] for g in genes])

    # If we have fewer samples than components, cap components
    actual_n_components = min(n_components, len(genes), X.shape[1])

    print(f"Reducing dimensions from {X.shape[1]} to {actual_n_components} via PCA...")
    pca = PCA(n_components=actual_n_components, random_state=42)
    X_reduced = pca.fit_transform(X)

    return {g: vec for g, vec in zip(genes, X_reduced)}

# ==========================================
# 3. VISUALIZATION: LATENT SPACE ANALYSIS
# ==========================================
def plot_latent_structure(embedding_dicts, gene_info, filename="latent_structure.png"):
    """Generates UMAPs for different embedding spaces."""
    num_models = len(embedding_dicts)
    fig, axes = plt.subplots(1, num_models, figsize=(3 * num_models, 3.5), sharex=True, sharey=True)
    if num_models == 1:
        axes = [axes]

    # --- Prepare data and color palette ---
    # Use top N gene types and group the rest as 'Other'
    top_n = 9
    top_types = gene_info['gene_type'].value_counts().nlargest(top_n).index

    # Create a color palette for top types + 'Other'
    palette = sns.color_palette("viridis", top_n)
    palette.append((0.8, 0.8, 0.8)) # Gray for 'Other'
    type_palette = dict(zip(top_types, palette))
    type_palette['Other'] = palette[-1]

    gene_info['plot_type'] = gene_info['gene_type'].apply(lambda x: x if x in top_types else 'Other')
    gene_to_type = dict(zip(gene_info['gene_name'], gene_info['plot_type']))

    print("Projecting Latent Spaces...")
    for ax, (name, embed_dict) in zip(axes, embedding_dicts.items()):
        # Use ALL genes in the embedding dictionary
        common_genes = [g for g in embed_dict.keys() if g in gene_to_type]
        total_genes = len(embed_dict)
        mapped_genes = len(common_genes)

        X = np.array([embed_dict[g] for g in common_genes])
        colors = [type_palette[gene_to_type.get(g, 'Other')] for g in common_genes]

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        proj = reducer.fit_transform(X)

        # Get gene descriptions for additional variation
        gene_to_desc = dict(zip(gene_info['gene_name'], gene_info.get('description', ['Unknown']*len(gene_info))))

        # Create shape markers based on description categories (e.g., first word of description)
        desc_categories = gene_info.get('description', pd.Series(['Unknown']*len(gene_info))).fillna('Unknown').str.split().str[0]
        top_desc_cats = desc_categories.value_counts().nlargest(5).index
        gene_info['desc_cat'] = desc_categories.apply(lambda x: x if x in top_desc_cats else 'Other')
        gene_to_desc_cat = dict(zip(gene_info['gene_name'], gene_info['desc_cat']))

        # Map description categories to marker shapes
        available_markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        unique_desc_cats = gene_info['desc_cat'].unique()
        marker_map = {cat: available_markers[i % len(available_markers)] for i, cat in enumerate(unique_desc_cats)}


        # Plot each description category separately to allow different markers
        for desc_cat in gene_info['desc_cat'].unique():
            mask = [gene_to_desc_cat.get(g, 'Other') == desc_cat for g in common_genes]
            if sum(mask) > 0:
                proj_subset = proj[mask]
                colors_subset = [colors[i] for i, m in enumerate(mask) if m]
                ax.scatter(proj_subset[:, 0], proj_subset[:, 1], c=colors_subset,
                           marker=marker_map.get(desc_cat, 'o'), s=0.5, alpha=0.6,
                           linewidth=0, rasterized=True)
        ax.set_title(f"{name}\n({mapped_genes}/{total_genes} genes)", weight='bold', fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

    # --- Create a shared legend ---
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                  markerfacecolor=color, markersize=5) for label, color in type_palette.items()]
    leg1 = fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.6), title='Gene Type', fontsize=7)

    marker_legend_elements = [plt.Line2D([0], [0], marker=marker, color='w', label=label,
                                         markerfacecolor='gray', markersize=5) for label, marker in marker_map.items()]
    fig.legend(handles=marker_legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.3), title='Description', fontsize=7)
    fig.add_artist(leg1)


    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()


def plot_latent_with_metadata(embedding_dicts, gene_info, metadata_file, columns_to_plot, base_filename="latent_metadata"):
    """
    Generates UMAPs for different embedding spaces, colored by different metadata attributes.

    Args:
        embedding_dicts (dict): Dictionary of embeddings.
        gene_info (pd.DataFrame): DataFrame with basic gene information.
        metadata_file (str): Path to the metadata CSV file.
        columns_to_plot (list): List of column names from metadata to use for coloring.
        base_filename (str): Base name for the output plot files.
    """
    # Load and merge metadata
    metadata_df = pd.read_csv(metadata_file)
    # Assuming 'gene_name' is the common column for merging
    merged_gene_info = pd.merge(gene_info, metadata_df, on='gene_name', how='left')

    for color_by_col in columns_to_plot:
        print(f"\nGenerating UMAPs colored by '{color_by_col}'...")

        # --- Prepare data and color palette ---
        unique_values = merged_gene_info[color_by_col].dropna().unique()

        # Use top N categories and group the rest as 'Other'
        top_n = 9
        if len(unique_values) > top_n + 1:
            top_categories = merged_gene_info[color_by_col].value_counts().nlargest(top_n).index
            merged_gene_info['plot_category'] = merged_gene_info[color_by_col].apply(lambda x: x if x in top_categories else 'Other')
            categories_to_map = list(top_categories) + ['Other']
        else:
            merged_gene_info['plot_category'] = merged_gene_info[color_by_col].fillna('Unknown')
            categories_to_map = list(merged_gene_info['plot_category'].unique())

        palette = sns.color_palette("viridis", len(categories_to_map) -1)
        if 'Other' in categories_to_map:
            palette.append((0.8, 0.8, 0.8)) # Gray for 'Other'
        if 'Unknown' in categories_to_map:
             palette.append((0.8, 0.8, 0.8)) # Gray for 'Unknown'

        category_palette = dict(zip(categories_to_map, palette))
        gene_to_category = dict(zip(merged_gene_info['gene_name'], merged_gene_info['plot_category']))

        # --- Create Plots ---
        num_models = len(embedding_dicts)
        fig, axes = plt.subplots(1, num_models, figsize=(2.5 * num_models + 1.5, 3), sharex=True, sharey=True)
        if num_models == 1:
            axes = [axes]

        for ax, (name, embed_dict) in zip(axes, embedding_dicts.items()):
            common_genes = [g for g in embed_dict.keys() if g in gene_to_category]
            X = np.array([embed_dict[g] for g in common_genes])
            colors = [category_palette.get(gene_to_category.get(g), category_palette.get('Other', (0.8,0.8,0.8))) for g in common_genes]

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            proj = reducer.fit_transform(X)

            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=0.5, alpha=0.6, linewidth=0, rasterized=True)
            ax.set_title(f"{name}", weight='bold', fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)

        # --- Create a shared legend ---
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                                      markerfacecolor=color, markersize=5) for label, color in category_palette.items()]
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.92, 0.5), title=color_by_col.replace('_', ' ').title(), fontsize=6)

        plt.suptitle(f"Latent Space Colored by {color_by_col.replace('_', ' ').title()}", fontsize=10, y=0.98)
        plt.subplots_adjust(left=0.02, right=0.88, top=0.88, bottom=0.05, wspace=0.1)

        filename = f"{base_filename}_{color_by_col}.png"
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved UMAP to {filename}")
        #plt.show()


def run_metadata_prediction_benchmark(embedding_dicts, gene_info, metadata_file):
    """
    Runs a benchmark to predict metadata attributes (chrom, start position) from gene embeddings.
    Fixes: Filters out classes with < 2 samples to prevent train_test_split errors.
    Returns: A list of dictionaries with results.
    """
    print("\n" + "="*40)
    print("Starting Metadata Prediction Benchmark")
    print("="*40)

    # Load and merge metadata
    metadata_df = pd.read_csv(metadata_file)
    merged_gene_info = pd.merge(gene_info, metadata_df, on='gene_name', how='inner')
    merged_gene_info = merged_gene_info.dropna(subset=['chrom', 'start'])

    # --- Task 1: Predict Chromosome ---
    print("\n--- Task: Predicting Chromosome ---")
    # Filter out rare chromosomes immediately (optional, but good for stability)
    chrom_counts = merged_gene_info['chrom'].value_counts()
    valid_chroms = chrom_counts[chrom_counts >= 2].index
    merged_gene_info_chrom = merged_gene_info[merged_gene_info['chrom'].isin(valid_chroms)].copy()

    le_chrom = LabelEncoder()
    merged_gene_info_chrom['chrom_label'] = le_chrom.fit_transform(merged_gene_info_chrom['chrom'])

    # --- Task 2: Predict Genomic Position (Binned) ---
    print("\n--- Task: Predicting Genomic Position (Binned) ---")
    merged_gene_info['start_bin'] = pd.qcut(merged_gene_info['start'], q=10, labels=False, duplicates='drop')
    le_bin = LabelEncoder()
    merged_gene_info['bin_label'] = le_bin.fit_transform(merged_gene_info['start_bin'])

    tasks = {
        'Chromosome': ('chrom_label', le_chrom, merged_gene_info_chrom),
        'Genomic Position (Binned)': ('bin_label', le_bin, merged_gene_info)
    }

    results = []
    for task_name, (label_col, encoder, source_df) in tasks.items():
        print(f"\n--- Evaluating for: {task_name} ---")

        for model_name, embed_dict in embedding_dicts.items():
            # 1. Prepare data intersection
            genes = [g for g in source_df['gene_name'] if g in embed_dict]
            if not genes:
                print(f"[{model_name}] No overlapping genes found. Skipping.")
                continue

            subset_df = source_df[source_df['gene_name'].isin(genes)]
            X = np.array([embed_dict[g] for g in subset_df['gene_name']])
            y = subset_df[label_col].values

            # 2. FILTER RARE CLASSES (The Fix)
            # Remove any class that has < 2 samples in the *current intersection*
            # This handles cases where a chromosome exists in the big table but
            # only 1 gene remains after intersecting with the embedding dictionary.
            class_counts = pd.Series(y).value_counts()
            valid_classes = class_counts[class_counts >= 2].index

            if len(valid_classes) < len(class_counts):
                # print(f"[{model_name}] Dropping {len(class_counts) - len(valid_classes)} rare classes (<2 samples).")
                mask = pd.Series(y).isin(valid_classes)
                X = X[mask]
                y = y[mask]

            if len(y) == 0:
                print(f"[{model_name}] No valid data after filtering rare classes.")
                continue

            # 3. Safe Split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                # Train Logistic Regression
                clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                print(f"[{model_name}] Accuracy for {task_name}: {accuracy:.3f}")
                results.append({
                    'Task': task_name,
                    'Model': model_name,
                    'Accuracy': accuracy
                })

            except ValueError as e:
                print(f"[{model_name}] Split failed despite filtering: {e}")
    return results


# ==========================================
# 4. RIGOROUS EVALUATION (BOOTSTRAPPED)
# ==========================================
def run_ggi_benchmark(embedding_dicts, train_df, test_df, n_bootstraps=50, filename="benchmark_summary.png", model_type='randomforest'):
    """Runs GGI task with ROC, PR, and Confidence Intervals.

    Args:
        embedding_dicts: Dictionary of embeddings to compare
        train_df: Training dataframe
        test_df: Test dataframe
        n_bootstraps: Number of bootstrap iterations
        filename: Output filename for the plot
        model_type: 'randomforest' or 'logistic' for classifier choice
    Returns:
        A list of dictionaries with model performance.
    """
    print(f"Starting Benchmark (n_bootstraps={n_bootstraps}, model={model_type})...")

    results = []
    plot_data = {'roc': {}, 'pr': {}}

    for name, embed_dict in embedding_dicts.items():
        X_train, y_train = build_ggi_features(train_df, embed_dict)
        X_test, y_test = build_ggi_features(test_df, embed_dict)

        # 1. Main Model Fit
        if model_type.lower() == 'logistic':
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model_name = "Logistic Regression"
        elif model_type.lower() == 'randomforest':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            model_name = "Random Forest"
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose 'randomforest' or 'logistic'.")

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        # Metrics for Curves
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        plot_data['roc'][name] = (fpr, tpr, auc(fpr, tpr))
        plot_data['pr'][name] = (rec, prec, average_precision_score(y_test, probs))

        # 2. Bootstrapping for Error Bars
        boot_scores = []
        for i in range(n_bootstraps):
            X_b, y_b = resample(X_test, y_test, random_state=i)
            b_probs = clf.predict_proba(X_b)[:, 1]
            boot_scores.append(roc_auc_score(y_b, b_probs))

        results.append({
            'Model': name,
            'AUC': auc(fpr, tpr),
            'AUC_std': np.std(boot_scores)
        })

    # --- FINAL FIGURE: PERFORMANCE SUMMARY ---
    res_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # A. ROC Curve
    for name, (fpr, tpr, score) in plot_data['roc'].items():
        axes[0].plot(fpr, tpr, color=COLORS.get(name, 'gray'), label=f'{name} ({score:.2f})', lw=1.2)
    axes[0].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axes[0].set_title(f'A. GGI Prediction (ROC) - {model_name}', loc='left', weight='bold')
    axes[0].legend(loc='lower right')

    # B. Comparative Bar Plot with Error Bars
    sns.barplot(data=res_df, x='Model', y='AUC', palette={model: COLORS.get(model, 'gray') for model in res_df['Model']}, ax=axes[1], capsize=.1)
    axes[1].errorbar(x=res_df['Model'], y=res_df['AUC'], yerr=res_df['AUC_std'], fmt='none', c='black', capsize=3, elinewidth=0.8)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title('B. Mean AUC (Bootstrapped)', loc='left', weight='bold')

    for ax in axes: sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    return results

def run_full_comparison(embedding_dicts, train_df, test_df, filename="full_roc_comparison.png", model_type='randomforest'):
    """Runs GGI task with ROC for all models including gene2vec and random.

    Args:
        embedding_dicts: Dictionary of embeddings to compare
        train_df: Training dataframe
        test_df: Test dataframe
        filename: Output filename for the plot
        model_type: 'randomforest' or 'logistic' for classifier choice
    Returns:
        A list of dictionaries with model performance.
    """
    print(f"Starting Full Comparison Benchmark with {model_type}...")

    # --- Load gene2vec embeddings ---
    if not os.path.exists(GENE2VEC_FILE):
        print(f"Warning: Gene2vec file not found at {GENE2VEC_FILE}. Skipping Gene2vec model.")
        gene2vec_embed = {}
    else:
        gene2vec_df = pd.read_csv(GENE2VEC_FILE, header=None, sep='\t')
        gene2vec_embed = {}
        for _, row in gene2vec_df.iterrows():
            gene_name = row[0]
            vector_str = row[1]
            vector = [float(x) for x in vector_str.split()]
            if len(vector) == 200:
                gene2vec_embed[gene_name] = np.array(vector)

    all_genes = set()
    if gene2vec_embed:
        all_genes.update(gene2vec_embed.keys())
    for d in embedding_dicts.values():
        all_genes.update(d.keys())

    # --- Create random embeddings ---
    np.random.seed(2023)
    random_embed = {gene: np.random.normal(size=1536) for gene in all_genes}

    # --- Combine all embeddings for evaluation ---
    full_embeddings = {
        **embedding_dicts,
    }
    if gene2vec_embed:
        full_embeddings['Gene2vec'] = gene2vec_embed
    full_embeddings['Random'] = random_embed


    plot_data = {}
    results = []

    for name, embed_dict in full_embeddings.items():
        print(f"Training and evaluating: {name}")
        X_train, y_train = build_ggi_features(train_df, embed_dict)
        X_test, y_test = build_ggi_features(test_df, embed_dict)

        if len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0:
            print(f"Skipping {name} due to no valid gene pairs in train/test set.")
            continue

        # --- Select model based on model_type ---
        if model_type.lower() == 'logistic':
            clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
            model_name = "Logistic Regression"
        elif model_type.lower() == 'randomforest':
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
            model_name = "Random Forest"
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose 'randomforest' or 'logistic'.")

        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        plot_data[name] = (fpr, tpr, score)
        results.append({'Model': name, 'AUC': score})

    # --- FINAL FIGURE: ROC CURVE COMPARISON ---
    plt.figure(figsize=(8, 8))

    # Define colors for all models
    full_colors = {
        **COLORS,
        'Gene2vec': '#8A2BE2', # purple
        'Random': '#A9A9A9' # dark gray
    }

    for name, (fpr, tpr, score) in plot_data.items():
        plt.plot(fpr, tpr, color=full_colors.get(name, 'gray'), lw=2, label=f'{name} (AUC = {score:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for Gene-Gene Interaction Prediction ({model_name})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()
    return results

# ==========================================
# 5. NEW BIOMEDICAL BENCHMARKS
# ==========================================
def run_go_term_prediction_benchmark(embedding_dicts, go_file):
    """
    Predicts GO terms from gene embeddings.
    Assumes go_file is a CSV with 'gene_name' and 'go_term' columns.
    """
    print("\n" + "="*40)
    print("Starting Gene Ontology (GO) Term Prediction Benchmark")
    print("="*40)

    if not os.path.exists(go_file):
        print(f"Warning: GO annotation file not found at {go_file}. Skipping benchmark.")
        return []

    go_df = pd.read_csv(go_file)
    # Create a list of GO terms for each gene
    gene_to_go = go_df.groupby('gene_name')['go_term'].apply(list).to_dict()

    results = []
    for model_name, embed_dict in embedding_dicts.items():
        # Prepare data
        common_genes = [g for g in gene_to_go.keys() if g in embed_dict]
        if len(common_genes) < 20: # Need enough data to split
            print(f"[{model_name}] Not enough overlapping genes with GO terms ({len(common_genes)}). Skipping.")
            continue

        X = np.array([embed_dict[g] for g in common_genes])
        y_labels = [gene_to_go[g] for g in common_genes]

        # Binarize labels
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y_labels)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a multi-label classifier
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Evaluate
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        print(f"[{model_name}] GO Prediction - Micro-F1: {micro_f1:.3f}, Macro-F1: {macro_f1:.3f}")
        results.append({'Model': model_name, 'Micro-F1': micro_f1, 'Macro-F1': macro_f1})

    return results


# ==========================================
# 6. EXECUTION
# ==========================================
def run_ablation_studies(embedding_dirs, base_output_dir="ablation_results"):
    """
    Run full analysis suite over a list of embedding directories for ablation studies.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    all_results = {
        'ggi_benchmark': [],
        'metadata_prediction': [],
        'full_comparison': [],
        'go_prediction': [],
        'clustering': [],
        'gene_analogy': []
    }

    # --- Load shared data once ---
    gene_info_table = pd.read_csv('./data/input_data/gene_info_table.csv')
    with open(f"./data/GenePT_gene_embedding_ada_text.pickle", "rb") as fp:
        gpt_3_5_gene_embeddings = pickle.load(fp)

    train_text_GGI = pd.read_csv('./data/predictionData/train_text.txt', sep=' ', header=None)
    train_label_GGI = pd.read_csv('./data/predictionData/train_label.txt', header=None)
    train_text_GGI.columns = ['gene_1', 'gene_2']
    train_label_GGI.columns = ['label']
    train_text_GGI_df = pd.concat([train_text_GGI, train_label_GGI], axis=1)

    test_text_GGI = pd.read_csv('./data/predictionData/test_text.txt', sep=' ', header=None)
    test_label_GGI = pd.read_csv('./data/predictionData/test_label.txt', header=None)
    test_text_GGI.columns = ['gene_1', 'gene_2']
    test_label_GGI.columns = ['label']
    test_text_GGI_df = pd.concat([test_text_GGI, test_label_GGI], axis=1)

    metadata_file = '../evo2/data/metadata.csv'

    # --- Loop through each embedding directory ---
    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Remap numpy._core to numpy.core for compatibility
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)

    def load_embedding(file_name):
        with open(os.path.join(embed_dir, file_name), "rb") as fp:
            return SafeUnpickler(fp).load()

    for embed_dir in embedding_dirs:
        print("\n" + "="*80)
        print(f"Processing directory: {embed_dir}")
        print("="*80)

        # --- Setup output directory and model name ---
        model_name_from_dir = os.path.basename(embed_dir)
        output_dir = os.path.join(base_output_dir, model_name_from_dir)
        os.makedirs(output_dir, exist_ok=True)

        # --- Load embeddings for the current directory ---
        try:

            # --- 1. Base Embeddings ---
            my_embeddings = {
                'GenePT (Original)': gpt_3_5_gene_embeddings,
                'GenePT (Proj)': load_embedding('dict_genept_proj.pkl'),
                'Evo2 (Proj)': load_embedding('dict_evo2_proj.pkl'), # Projected
                'Evo2 (Mean)': load_embedding('dict_mean_evo2.pkl'),
            }

            # Helper to safely get dimensions
            def get_dim(d):
                if not d: return 0
                return next(iter(d.values())).shape[0]

            dim_gpt_orig = 1536
            dim_evo2_mean = 1920

            # --- 2. Concatenations ---
            genept_concat = concat_embeddings(my_embeddings['GenePT (Proj)'], my_embeddings['GenePT (Original)'])
            if genept_concat:
                my_embeddings['GenePT Concat (Proj + Original)'] = genept_concat
                my_embeddings['GenePT Concat + PCA'] = reduce_dimensions(genept_concat, dim_gpt_orig)

            del genept_concat
            gc.collect()

            # Evo2 Concat (Proj + Mean)
            evo2_concat = concat_embeddings(my_embeddings['Evo2 (Proj)'], my_embeddings['Evo2 (Mean)'])
            if evo2_concat:
                my_embeddings['Evo2 Concat (Proj + Mean)'] = evo2_concat
                my_embeddings['Evo2 Concat + PCA'] = reduce_dimensions(evo2_concat, dim_evo2_mean)

            del evo2_concat
            gc.collect()

            # Cross Concat (Evo2 Proj + GenePT Proj)
            cross_proj_concat = concat_embeddings(my_embeddings['Evo2 (Proj)'], my_embeddings['GenePT (Proj)'])
            if cross_proj_concat:
                my_embeddings['Cross Concat (Evo2 Proj + GenePT Proj)'] = cross_proj_concat

            del cross_proj_concat
            gc.collect()

            # Original Concat (Evo2 Mean + GenePT Original)
            orig_concat = concat_embeddings(my_embeddings['Evo2 (Mean)'], my_embeddings['GenePT (Original)'])
            if orig_concat:
                my_embeddings['Original Concat (Evo2 Mean + GenePT Original)'] = orig_concat

            del orig_concat
            gc.collect()
        except FileNotFoundError as e:
            print(f"Skipping directory {embed_dir} due to missing file: {e}")
            continue


        # --- Run benchmarks and collect results ---

        # 1. GGI Prediction Benchmark with ROC and PR curves
        ggi_results = run_ggi_benchmark(my_embeddings, train_text_GGI_df, test_text_GGI_df, filename=os.path.join(output_dir, "benchmark_summary.png"), model_type='logistic')
        for res in ggi_results:
            res['Experiment'] = model_name_from_dir
        all_results['ggi_benchmark'].extend(ggi_results)

        full_comp_results = run_full_comparison(my_embeddings, train_text_GGI_df, test_text_GGI_df, filename=os.path.join(output_dir, "full_roc_comparison.png"), model_type='logistic')
        for res in full_comp_results:
            res['Experiment'] = model_name_from_dir
        all_results['full_comparison'].extend(full_comp_results)


        # 2. Produce latent embedding plots
        if os.path.exists(metadata_file):
            plot_latent_with_metadata(
                my_embeddings, gene_info_table, metadata_file,
                columns_to_plot=['chrom', 'gene_type_x', 'strand'],
                base_filename=os.path.join(output_dir, "latent_structure_metadata")
            )
            meta_results = run_metadata_prediction_benchmark(my_embeddings, gene_info_table, metadata_file)
            for res in meta_results:
                res['Experiment'] = model_name_from_dir
            all_results['metadata_prediction'].extend(meta_results)
        else:
            print(f"Metadata file not found at {metadata_file}, skipping metadata plots and benchmark.")


        # 3. TODO: Additional downstream tasks from GenePT ...


    # --- Aggregate and save all results ---
    if all_results['ggi_benchmark']:
        ggi_df = pd.DataFrame(all_results['ggi_benchmark'])
        ggi_df.to_csv(os.path.join(base_output_dir, "ggi_benchmark_results.csv"), index=False)
        print("\n--- GGI Benchmark Summary ---")
        print(ggi_df)

    if all_results['metadata_prediction']:
        meta_df = pd.DataFrame(all_results['metadata_prediction'])
        meta_df.to_csv(os.path.join(base_output_dir, "metadata_prediction_results.csv"), index=False)
        print("\n--- Metadata Prediction Summary ---")
        print(meta_df)

    if all_results['full_comparison']:
        full_df = pd.DataFrame(all_results['full_comparison'])
        full_df.to_csv(os.path.join(base_output_dir, "full_comparison_results.csv"), index=False)
        print("\n--- Full Comparison Summary ---")
        print(full_df)


    print(f"\nAblation study complete. All results saved in '{base_output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation studies on gene embeddings")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Specific directory name in the processed folder. If not provided, all directories are processed."
    )
    args = parser.parse_args()
    specified_dir = args.dir

    processed_root = "../evo2/data/output/processed"

    if specified_dir:
        ablation_dir = os.path.join(processed_root, specified_dir)
        if not os.path.isdir(ablation_dir):
            print(f"Error: Specified directory '{specified_dir}' not found in '{processed_root}'.")
            exit(1)
        ABLATION_DIRS = [ablation_dir]
    else:
        ABLATION_DIRS = sorted(
            os.path.join(processed_root, d)
            for d in os.listdir(processed_root)
            if os.path.isdir(os.path.join(processed_root, d))
        )

    run_ablation_studies(ABLATION_DIRS)
