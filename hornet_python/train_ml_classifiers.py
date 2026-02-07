"""
Train Traditional ML Classifiers
Alternative approach to the geometric polygon classification.

This script trains various ML classifiers on the 2D FT training data:
- Linear Discriminant Analysis (LDA) with .predict()
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Neural Network (MLP)

The trained model can then be used for direct classification without
polygon boundaries.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_training_data():
    """
    Load and prepare training data from fourth_TDB.pkl

    Returns:
    --------
    X : array, shape (n_samples, n_features)
        Feature vectors (flattened 2D FT spectra)
    y : array, shape (n_samples,)
        Class labels (0=hornet, 1=bee, 2=winter bg, 3=summer bg)
    class_names : list
        Human-readable class names
    """
    print("Loading training database...")
    with open('fourth_TDB.pkl', 'rb') as f:
        tdb_data = pickle.load(f)

    scaled_hornet = tdb_data['scaled_hornet']
    scaled_bee = tdb_data['scaled_bee']
    scaled_BG = tdb_data['scaled_BG']
    scaled_BGE = tdb_data['scaled_BGE']

    # Reshape to vectors
    def reshape_to_vectors(data_3d):
        n_samples = data_3d.shape[2]
        data_list = []
        for i in range(n_samples):
            spectrum = data_3d[:, :, i]
            flattened = spectrum.flatten()
            data_list.append(flattened)
        return np.array(data_list)

    hornet_array = reshape_to_vectors(scaled_hornet)
    bee_array = reshape_to_vectors(scaled_bee)
    BG_array = reshape_to_vectors(scaled_BG)
    summerBG_array = reshape_to_vectors(scaled_BGE)

    # Combine all data
    X = np.vstack([hornet_array, bee_array, BG_array, summerBG_array])

    # Create labels
    n_hornet = hornet_array.shape[0]
    n_bee = bee_array.shape[0]
    n_bg = BG_array.shape[0]
    n_bgs = summerBG_array.shape[0]

    y = np.array([0] * n_hornet + [1] * n_bee + [2] * n_bg + [3] * n_bgs)

    class_names = ['Hornet', 'Bee', 'Winter BG', 'Summer BG']

    print(f"  Total samples: {len(y)}")
    print(f"  Hornet: {n_hornet}")
    print(f"  Bee: {n_bee}")
    print(f"  Winter background: {n_bg}")
    print(f"  Summer background: {n_bgs}")
    print(f"  Features per sample: {X.shape[1]}")

    return X, y, class_names


def train_and_evaluate_classifiers(X, y, class_names):
    """
    Train and evaluate multiple ML classifiers using cross-validation.

    Parameters:
    -----------
    X : array
        Feature matrix
    y : array
        Labels
    class_names : list
        Class names

    Returns:
    --------
    results : dict
        Dictionary of trained models and their scores
    """
    print("\n" + "=" * 60)
    print("Training and Evaluating Classifiers")
    print("=" * 60)

    # Preprocessing pipeline with PCA for dimensionality reduction
    # Original approach uses 9 PCA components for DFA
    # We'll use a similar approach to reduce the 2744 features
    print(f"\nOriginal features: {X.shape[1]}")
    print("Applying PCA to reduce dimensionality (keeps 95% variance)...")

    scaler = StandardScaler()
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance

    X_scaled = scaler.fit_transform(X)
    X_reduced = pca.fit_transform(X_scaled)

    print(f"Reduced features: {X_reduced.shape[1]} (from {X.shape[1]})")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Define classifiers
    # Note: Using HistGradientBoostingClassifier which is much faster than GradientBoostingClassifier
    classifiers = {
        'LDA': LinearDiscriminantAnalysis(),
        'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=50, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, clf in classifiers.items():
        print(f"\n{name}:")
        print("-" * 40)

        # Add progress indicator for slower methods
        if 'Gradient' in name or 'Neural' in name:
            print("  Training (this may take a moment)...")

        # Perform cross-validation
        scores = cross_val_score(clf, X_reduced, y, cv=cv, scoring='accuracy', n_jobs=1)

        print(f"  Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

        # Train on full dataset
        clf.fit(X_reduced, y)
        train_score = clf.score(X_reduced, y)
        print(f"  Training accuracy: {train_score:.3f}")

        # Get predictions for confusion matrix
        y_pred = clf.predict(X_reduced)

        # Store results
        results[name] = {
            'classifier': clf,
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'train_score': train_score,
            'y_pred': y_pred
        }

    # Store preprocessing for later use
    results['scaler'] = scaler
    results['pca'] = pca

    return results


def visualize_results(results, y, class_names):
    """
    Create visualizations comparing classifier performance.

    Parameters:
    -----------
    results : dict
        Results from train_and_evaluate_classifiers
    y : array
        True labels
    class_names : list
        Class names
    """
    print("\n" + "=" * 60)
    print("Generating Visualizations")
    print("=" * 60)

    # 1. Comparison bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter out preprocessing objects (scaler, pca)
    classifier_names = [name for name in results.keys()
                        if name not in ['scaler', 'pca']]
    cv_means = [results[name]['cv_mean'] for name in classifier_names]
    cv_stds = [results[name]['cv_std'] for name in classifier_names]

    x_pos = np.arange(len(classifier_names))
    ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xlabel('Classifier', fontsize=12)
    ax.set_ylabel('Cross-validation Accuracy', fontsize=12)
    ax.set_title('Classifier Performance Comparison', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classifier_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('classifier_comparison.png', dpi=150)
    print("  Saved classifier_comparison.png")

    # 2. Find best classifier
    best_classifier_name = max(classifier_names, key=lambda x: results[x]['cv_mean'])
    best_result = results[best_classifier_name]

    print(f"\nBest classifier: {best_classifier_name}")
    print(f"  CV Accuracy: {best_result['cv_mean']:.3f} (+/- {best_result['cv_std']:.3f})")

    # 3. Confusion matrix for best classifier
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y, best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {best_classifier_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix_best.png', dpi=150)
    print("  Saved confusion_matrix_best.png")

    # 4. Classification report
    print("\n" + "=" * 60)
    print(f"Classification Report - {best_classifier_name}")
    print("=" * 60)
    print(classification_report(y, best_result['y_pred'], target_names=class_names))

    return best_classifier_name


def save_best_model(results, best_name, X, y):
    """
    Save the best trained model for later use.

    Parameters:
    -----------
    results : dict
        Training results
    best_name : str
        Name of best classifier
    X : array
        Feature matrix
    y : array
        Labels
    """
    print("\n" + "=" * 60)
    print("Saving Best Model")
    print("=" * 60)

    best_classifier = results[best_name]['classifier']
    scaler = results['scaler']
    pca = results['pca']

    # Get feature importance if available
    feature_importance = None
    if hasattr(best_classifier, 'feature_importances_'):
        feature_importance = best_classifier.feature_importances_

    model_data = {
        'classifier': best_classifier,
        'classifier_name': best_name,
        'scaler': scaler,
        'pca': pca,
        'cv_accuracy': results[best_name]['cv_mean'],
        'cv_std': results[best_name]['cv_std'],
        'train_accuracy': results[best_name]['train_score'],
        'feature_importance': feature_importance,
        'n_features_original': X.shape[1],
        'n_features_reduced': pca.n_components_,
        'n_classes': len(np.unique(y)),
        'class_names': ['Hornet', 'Bee', 'Winter BG', 'Summer BG']
    }

    output_file = 'trained_ml_classifier.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"  Model saved to {output_file}")
    print(f"  Classifier: {best_name}")
    print(f"  Features: {model_data['n_features_original']} → {model_data['n_features_reduced']} (PCA)")
    print(f"  CV Accuracy: {model_data['cv_accuracy']:.3f} (+/- {model_data['cv_std']:.3f})")
    print(f"  Training Accuracy: {model_data['train_accuracy']:.3f}")


def main():
    """
    Main function to train and evaluate ML classifiers.
    """
    print("=" * 60)
    print("Traditional ML Classifier Training")
    print("=" * 60)

    # Load training data
    X, y, class_names = load_training_data()

    # Train and evaluate classifiers
    results = train_and_evaluate_classifiers(X, y, class_names)

    # Visualize results
    best_name = visualize_results(results, y, class_names)

    # Save best model
    save_best_model(results, best_name, X, y)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Use 'classify_with_ml.py' to classify new audio files")
    print("  2. Compare with polygon-based classification results")


if __name__ == "__main__":
    main()
