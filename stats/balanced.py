import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import random

class BalancedComparison:
    def __init__(self, random_state=42):
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
        
    def create_balanced_dataset(self, human_segments, llm_segments, balance_method='undersample'):
        """
        Create balanced dataset using different strategies
        
        Args:
            human_segments: List of human text segments
            llm_segments: List of LLM text segments  
            balance_method: 'undersample', 'oversample', or 'fixed_size'
        """
        print(f"Original sizes: {len(human_segments)} human, {len(llm_segments)} LLM")
        
        if balance_method == 'undersample':
            # Undersample majority class (human) to match minority class (LLM)
            min_size = min(len(human_segments), len(llm_segments))
            balanced_human = random.sample(human_segments, min_size)
            balanced_llm = llm_segments[:min_size]  # Take all LLM if it's smaller
            
        elif balance_method == 'oversample':
            # Oversample minority class (LLM) to match majority class (human)
            max_size = max(len(human_segments), len(llm_segments))
            balanced_human = human_segments
            # Repeat LLM segments to reach target size
            balanced_llm = []
            while len(balanced_llm) < max_size:
                balanced_llm.extend(llm_segments)
            balanced_llm = balanced_llm[:max_size]
            
        elif balance_method == 'fixed_size':
            # Use fixed size (e.g., 10,000 each) for computational efficiency
            target_size = min(10000, len(human_segments), len(llm_segments))
            balanced_human = random.sample(human_segments, target_size)
            if len(llm_segments) >= target_size:
                balanced_llm = random.sample(llm_segments, target_size)
            else:
                # Oversample if needed
                balanced_llm = llm_segments * (target_size // len(llm_segments) + 1)
                balanced_llm = balanced_llm[:target_size]
        
        print(f"Balanced sizes: {len(balanced_human)} human, {len(balanced_llm)} LLM")
        return balanced_human, balanced_llm
    
    def compare_methods(self, human_segments, llm_segments, balance_methods=['undersample', 'fixed_size']):
        """
        Compare different balancing methods
        """
        results = {}
        
        for method in balance_methods:
            print(f"\n=== TESTING {method.upper()} METHOD ===")
            
            # Create balanced dataset
            balanced_human, balanced_llm = self.create_balanced_dataset(
                human_segments, llm_segments, method
            )
            
            # Prepare data
            all_segments = balanced_human + balanced_llm
            all_labels = [0] * len(balanced_human) + [1] * len(balanced_llm)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                all_segments, all_labels, test_size=0.2, 
                random_state=self.random_state, stratify=all_labels
            )
            
            # Train classifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Store results
            results[method] = {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': classification_report(y_test, y_pred, target_names=['Human', 'LLM']),
                'human_recall': cm[0,0] / (cm[0,0] + cm[0,1]),  # True negatives / (TN + FP)
                'llm_recall': cm[1,1] / (cm[1,0] + cm[1,1]),    # True positives / (FN + TP)
                'human_precision': cm[0,0] / (cm[0,0] + cm[1,0]),  # TN / (TN + FN)
                'llm_precision': cm[1,1] / (cm[0,1] + cm[1,1]),    # TP / (FP + TP)
                'pipeline': pipeline
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Human recall: {results[method]['human_recall']:.4f}")
            print(f"LLM recall: {results[method]['llm_recall']:.4f}")
            
        return results
    
    def cross_validate_balanced(self, human_segments, llm_segments, balance_method='undersample', cv_folds=5):
        """
        Perform cross-validation with balanced datasets
        """
        # Create balanced dataset
        balanced_human, balanced_llm = self.create_balanced_dataset(
            human_segments, llm_segments, balance_method
        )
        
        all_segments = balanced_human + balanced_llm
        all_labels = [0] * len(balanced_human) + [1] * len(balanced_llm)
        
        # Stratified K-fold to maintain class balance
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {
            'accuracy': [],
            'human_recall': [],
            'llm_recall': [],
            'human_precision': [],
            'llm_precision': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(all_segments, all_labels)):
            X_train = [all_segments[i] for i in train_idx]
            X_test = [all_segments[i] for i in test_idx]
            y_train = [all_labels[i] for i in train_idx]
            y_test = [all_labels[i] for i in test_idx]
            
            # Train classifier
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            cv_results['accuracy'].append(accuracy)
            cv_results['human_recall'].append(cm[0,0] / (cm[0,0] + cm[0,1]))
            cv_results['llm_recall'].append(cm[1,1] / (cm[1,0] + cm[1,1]))
            cv_results['human_precision'].append(cm[0,0] / (cm[0,0] + cm[1,0]))
            cv_results['llm_precision'].append(cm[1,1] / (cm[0,1] + cm[1,1]))
            
            print(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Human recall={cv_results['human_recall'][-1]:.4f}, LLM recall={cv_results['llm_recall'][-1]:.4f}")
        
        # Calculate means and stds
        cv_summary = {}
        for metric in cv_results:
            cv_summary[metric] = {
                'mean': np.mean(cv_results[metric]),
                'std': np.std(cv_results[metric])
            }
        
        return cv_summary, cv_results
    
    def plot_comparison(self, results, ais_results=None):
        """
        Plot comparison between different methods and optionally AIS results
        """
        methods = list(results.keys())
        if ais_results:
            methods.extend(['AIS Random', 'AIS Greedy', 'AIS Genetic'])
        
        # Prepare data for plotting
        human_recalls = [results[m]['human_recall'] for m in list(results.keys())]
        llm_recalls = [results[m]['llm_recall'] for m in list(results.keys())]
        
        if ais_results:
            # Add AIS results (convert percentage to decimal)
            human_recalls.extend([ais_results['random']/100, ais_results['greedy']/100, ais_results['genetic']/100])
            llm_recalls.extend([0.0, 0.0, 0.0])  # AIS doesn't measure LLM recall directly
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Human recall comparison
        bars1 = ax1.bar(methods, human_recalls, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
        ax1.set_ylabel('Human Detection Rate')
        ax1.set_title('Human Text Detection Comparison')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, human_recalls):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # LLM recall comparison (only for ML methods)
        ml_methods = list(results.keys())
        ml_llm_recalls = [results[m]['llm_recall'] for m in ml_methods]
        
        bars2 = ax2.bar(ml_methods, ml_llm_recalls, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax2.set_ylabel('LLM Detection Rate')
        ax2.set_title('LLM Text Detection (ML Methods Only)')
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars2, ml_llm_recalls):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage with your data
def main():
    print("=== BALANCED COMPARISON: AIS vs LINEAR REGRESSION ===")
    
    # Initialize comparison
    comparator = BalancedComparison(random_state=42)
    
    # Your AIS results for reference
    ais_results = {
        'random': 46.0,
        'greedy': 68.9, 
        'genetic': 53.4
    }
    
    # Load your data (replace with your actual file paths)
    def load_lines(file, max_lines=None):
        with open(file) as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines[:max_lines] if max_lines else lines

    def prepare_segments(lines, segment_length=6):
        segments = []
        for line in lines:
            words = line.split()
            if len(words) >= segment_length:
                for i in range(len(words) - segment_length + 1):
                    segment = ' '.join(words[i:i + segment_length])
                    segments.append(segment)
        return segments
    
    # File paths (update these to your actual paths)
    HUMAN_FILE = "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/train_human_clean.txt"
    LLM_FILE = "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/test_llm_clean.txt"
    
    try:
        # Load data
        human_lines = load_lines(HUMAN_FILE)
        llm_lines = load_lines(LLM_FILE)
        
        # Create segments
        human_segments = prepare_segments(human_lines)
        llm_segments = prepare_segments(llm_lines)
        
        print(f"Created {len(human_segments)} human segments and {len(llm_segments)} LLM segments")
        
        # Compare different balancing methods
        results = comparator.compare_methods(human_segments, llm_segments, 
                                           balance_methods=['undersample', 'fixed_size'])
        
        # Cross-validation with balanced data
        print(f"\n=== CROSS-VALIDATION (BALANCED) ===")
        cv_summary, cv_results = comparator.cross_validate_balanced(
            human_segments, llm_segments, balance_method='undersample', cv_folds=5
        )
        
        print(f"\nCross-validation summary (undersample method):")
        for metric, stats in cv_summary.items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Plot comparison
        print(f"\n=== COMPARISON WITH AIS RESULTS ===")
        fig = comparator.plot_comparison(results, ais_results)
        
        # Summary for report
        print(f"\n" + "="*60)
        print("BALANCED COMPARISON SUMMARY")
        print("="*60)
        
        for method, result in results.items():
            print(f"\n{method.upper()} METHOD:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Human recall: {result['human_recall']:.4f}")
            print(f"  LLM recall: {result['llm_recall']:.4f}")
            print(f"  Human precision: {result['human_precision']:.4f}")
            print(f"  LLM precision: {result['llm_precision']:.4f}")
        
        print(f"\nAIS RESULTS (for comparison):")
        print(f"  Random: {ais_results['random']:.1f}% human detection")
        print(f"  Greedy: {ais_results['greedy']:.1f}% human detection")
        print(f"  Genetic: {ais_results['genetic']:.1f}% human detection")
        
        return results, cv_summary, ais_results
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None

if __name__ == "__main__":
    results, cv_summary, ais_results = main()