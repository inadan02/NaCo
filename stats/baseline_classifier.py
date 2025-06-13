import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import re

class BaselineClassifier:
    def __init__(self):
        self.pipeline = None
        self.results = {}
        
    def preprocess_text(self, text):
        """
        Apply the same preprocessing as your AIS model
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def load_and_prepare_data(self, human_texts, llm_texts):
        """
        Prepare the dataset for classification
        
        Args:
            human_texts: List of human-written text segments
            llm_texts: List of LLM-generated text segments
        """
        # Preprocess texts
        human_processed = [self.preprocess_text(text) for text in human_texts]
        llm_processed = [self.preprocess_text(text) for text in llm_texts]
        
        # Create labels (0 for human, 1 for LLM)
        human_labels = [0] * len(human_processed)
        llm_labels = [1] * len(llm_processed)
        
        # Combine data
        all_texts = human_processed + llm_processed
        all_labels = human_labels + llm_labels
        
        return all_texts, all_labels
    
    def create_segments(self, texts, segment_length=6):
        """
        Create 6-word segments like your AIS model
        """
        segments = []
        for text in texts:
            words = text.split()
            if len(words) >= segment_length:
                for i in range(len(words) - segment_length + 1):
                    segment = ' '.join(words[i:i + segment_length])
                    segments.append(segment)
        return segments
    
    def train_classifier(self, texts, labels, test_size=0.2, random_state=42):
        """
        Train the baseline classifier
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Create pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,  # Limit features to avoid overfitting
                ngram_range=(1, 2),  # Use unigrams and bigrams
                min_df=2,           # Ignore terms that appear in < 2 documents
                max_df=0.95         # Ignore terms that appear in > 95% of documents
            )),
            ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
        ])
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)
        
        # Store results
        self.results = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, target_names=['Human', 'LLM']),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return self.results
    
    def cross_validate(self, texts, labels, cv=5):
        """
        Perform cross-validation
        """
        if self.pipeline is None:
            # Create pipeline if not exists
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
        
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }
    
    def analyze_features(self, top_n=20):
        """
        Analyze the most important features for classification
        """
        if self.pipeline is None:
            print("Model not trained yet!")
            return
        
        # Get feature names and coefficients
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.pipeline.named_steps['classifier'].coef_[0]
        
        # Get top features for each class
        feature_importance = list(zip(feature_names, coefficients))
        
        # Sort by coefficient (positive = LLM-like, negative = Human-like)
        feature_importance.sort(key=lambda x: x[1])
        
        print("Top features indicating HUMAN text:")
        for feature, coef in feature_importance[:top_n]:
            print(f"  {feature}: {coef:.4f}")
        
        print(f"\nTop features indicating LLM text:")
        for feature, coef in feature_importance[-top_n:]:
            print(f"  {feature}: {coef:.4f}")
        
        return feature_importance
    
    def plot_results(self):
        """
        Create visualizations of the results
        """
        if not self.results:
            print("No results to plot. Train the model first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion Matrix
        sns.heatmap(self.results['confusion_matrix'], 
                   annot=True, fmt='d', 
                   xticklabels=['Human', 'LLM'], 
                   yticklabels=['Human', 'LLM'],
                   ax=axes[0,0])
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_ylabel('True Label')
        axes[0,0].set_xlabel('Predicted Label')
        
        # Prediction probabilities distribution
        human_probs = self.results['y_pred_proba'][self.results['y_test'] == 0][:, 1]
        llm_probs = self.results['y_pred_proba'][self.results['y_test'] == 1][:, 1]
        
        axes[0,1].hist(human_probs, alpha=0.5, label='Human texts', bins=20)
        axes[0,1].hist(llm_probs, alpha=0.5, label='LLM texts', bins=20)
        axes[0,1].set_xlabel('Probability of being LLM')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Prediction Probability Distribution')
        axes[0,1].legend()
        
        # Accuracy by confidence
        probs = self.results['y_pred_proba'].max(axis=1)
        correct = (self.results['y_pred'] == self.results['y_test'])
        
        # Bin by confidence
        confidence_bins = np.linspace(0.5, 1.0, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins)-1):
            mask = (probs >= confidence_bins[i]) & (probs < confidence_bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        axes[1,0].bar(range(len(bin_accuracies)), bin_accuracies, alpha=0.7)
        axes[1,0].set_xlabel('Confidence Bin')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Accuracy by Confidence Level')
        axes[1,0].set_xticks(range(len(bin_accuracies)))
        axes[1,0].set_xticklabels([f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}' 
                                  for i in range(len(confidence_bins)-1)], rotation=45)
        
        # Sample count by confidence
        axes[1,1].bar(range(len(bin_counts)), bin_counts, alpha=0.7, color='orange')
        axes[1,1].set_xlabel('Confidence Bin')
        axes[1,1].set_ylabel('Number of Samples')
        axes[1,1].set_title('Sample Distribution by Confidence')
        axes[1,1].set_xticks(range(len(bin_counts)))
        axes[1,1].set_xticklabels([f'{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}' 
                                  for i in range(len(confidence_bins)-1)], rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """
        Print a summary of the results
        """
        if not self.results:
            print("No results available. Train the model first!")
            return
        
        print("=== BASELINE CLASSIFIER RESULTS ===")
        print(f"Accuracy: {self.results['accuracy']:.4f}")
        print("\nClassification Report:")
        print(self.results['classification_report'])
        print(f"\nConfusion Matrix:")
        print(self.results['confusion_matrix'])

# === DATA LOADING (matching your GA setup) ===
def load_lines(file, max_lines=None):
    """Load lines from file, same as your GA code"""
    with open(file) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:max_lines] if max_lines else lines

def prepare_segments_with_labels(human_lines, llm_lines, segment_length=6):
    """
    Create 6-word segments and their labels, matching your AIS approach
    """
    all_segments = []
    all_labels = []
    
    # Process human text (label = 0)
    for line in human_lines:
        words = line.split()
        if len(words) >= segment_length:
            for i in range(len(words) - segment_length + 1):
                segment = ' '.join(words[i:i + segment_length])
                all_segments.append(segment)
                all_labels.append(0)  # 0 = human
    
    # Process LLM text (label = 1) 
    for line in llm_lines:
        words = line.split()
        if len(words) >= segment_length:
            for i in range(len(words) - segment_length + 1):
                segment = ' '.join(words[i:i + segment_length])
                all_segments.append(segment)
                all_labels.append(1)  # 1 = LLM
                
    return all_segments, all_labels

# Example usage matching your project setup:
def main():
    # File paths from your GA code
    LLM_FILE = "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/test_llm_clean.txt"
    HUMAN_FILE = "/Users/anilayhan/Desktop/Natural Computing/project last/NaCo/our_data/train_human_clean.txt"
    
    print("=== BASELINE CLASSIFIER FOR HUMAN vs LLM TEXT ===")
    print(f"Loading data from:")
    print(f"  Human text: {HUMAN_FILE}")
    print(f"  LLM text: {LLM_FILE}")
    
    # Load data using your format
    try:
        human_lines = load_lines(HUMAN_FILE)
        llm_lines = load_lines(LLM_FILE)
        print(f"Loaded {len(human_lines)} human lines and {len(llm_lines)} LLM lines")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please make sure the data files exist in the correct paths")
        return
    
    # Create segments with labels (same as your AIS approach)
    segments, labels = prepare_segments_with_labels(human_lines, llm_lines, segment_length=6)
    print(f"Created {len(segments)} total segments ({labels.count(0)} human, {labels.count(1)} LLM)")
    
    # Initialize classifier
    classifier = BaselineClassifier()
    
    # Train and evaluate
    print("\n=== TRAINING CLASSIFIER ===")
    results = classifier.train_classifier(segments, labels)
    
    print(f"\nBaseline Classifier Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Cross-validation
    print("\n=== CROSS-VALIDATION ===")
    cv_results = classifier.cross_validate(segments, labels)
    print(f"Cross-validation accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']*2:.4f})")
    
    # Feature analysis
    print("\n=== FEATURE ANALYSIS ===")
    classifier.analyze_features(top_n=15)
    
    # Plot results
    print("\n=== GENERATING PLOTS ===")
    classifier.plot_results()
    
    # Summary for your report
    print("\n" + "="*50)
    print("SUMMARY FOR REPORT:")
    print("="*50)
    print(f"Dataset: {len(human_lines)} human lines, {len(llm_lines)} LLM lines")
    print(f"Segments: {len(segments)} total 6-word segments")
    print(f"Baseline Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']*2:.4f}")
    
    # Compare with your AIS results
    human_correct = results['confusion_matrix'][0,0]  # True negatives
    llm_correct = results['confusion_matrix'][1,1]    # True positives
    total_human = results['confusion_matrix'][0,0] + results['confusion_matrix'][0,1]
    total_llm = results['confusion_matrix'][1,0] + results['confusion_matrix'][1,1]
    
    print(f"Human detection rate: {human_correct/total_human:.4f}")
    print(f"LLM detection rate: {llm_correct/total_llm:.4f}")
    print("\nThis can be compared with your AIS reactivity scores!")
    
    return classifier, results, cv_results

if __name__ == "__main__":
    main()
    print("yo")