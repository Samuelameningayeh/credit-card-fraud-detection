import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc
)
from imblearn.under_sampling import RandomUnderSampler

class FraudReplicationStudy:
    def __init__(self, data_path):
        """
        Replicates the methodology from Ayeh et al. (2023).
        """
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Define the 3 algorithms from the paper
        self.models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1) 
        }
        self.results = []

    def preprocess(self):
        """
        Cleaning steps based on the paper's description.
        """
        print(f"Original Data Shape: {self.df.shape}")
        
        # 1. Filter Amount: "Values less than 5 and greater than 1250 were removed"
        # Note: Using 'amt' as the standard column name for this dataset
        if 'amt' in self.df.columns:
             initial_rows = len(self.df)
             self.df = self.df[(self.df['amt'] >= 5) & (self.df['amt'] <= 1250)]
             print(f"Rows dropped due to Amount filtering: {initial_rows - len(self.df)}")

        # 2. Drop High Correlation & Irrelevant Columns
        # "Merchant longitude and longitude variables were highly correlated... we manually removed one"
        # Also dropping PII (names) and high cardinality columns
        drop_cols = ['Unnamed: 0', 'trans_num', 'trans_date_trans_time', 
                     'first', 'last', 'street', 'city', 'state', 'dob', 
                     'merch_long', 'merch_lat', 'zip', 'job', 'merchant']
        
        self.df = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns], errors='ignore')
        
        # 3. Encoding Categoricals
        self.df = pd.get_dummies(self.df, drop_first=True)
        
        return self.df

    def run_analysis(self):
        # Step 1: Clean Data
        self.preprocess()

        # Step 2: Prepare X and y
        target = 'is_fraud'
        X = self.df.drop(target, axis=1)
        y = self.df[target]

        # Step 3: Scale Data (0 to 1 range transformation)
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        # Step 4: Split Data
        # Splitting BEFORE undersampling to ensure the Test set remains realistic (imbalanced)
        print("Splitting Data into Train (70%) and Test (30%)...")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Step 5: Handle Imbalance (Training Set ONLY)
        # "We employed under sampling to handle the imbalance"
        print("Under-sampling the Training Data...")
        rus = RandomUnderSampler(random_state=42)
        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
        print(f"Training with {len(X_train_res)} balanced samples.")

        # Step 6: Train and Evaluate Models
        plt.figure(figsize=(10, 8))
        
        print("\n--- Model Results ---")
        for name, model in self.models.items():
            # Train
            model.fit(X_train_res, y_train_res)
            
            # Predict on the (Imbalanced) Test Set
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate Metrics
            # Specificity = TN / (TN + FP)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp)
            
            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Sensitivity (Recall)": recall_score(y_test, y_pred),
                "Specificity": specificity,
                "F1-Score": f1_score(y_test, y_pred),
                "AUC": auc(*roc_curve(y_test, y_prob)[:2])
            }
            self.results.append(metrics)
            
            # Plot ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['AUC']:.3f})")

        # Finalize Plot
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.title('Model Comparison: ROC Curves')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend()
        plt.savefig('roc_curves.png')
        plt.show()
    
        
        # Display Comparison Table
        results_df = pd.DataFrame(self.results).set_index("Model")
        print("\nSummary Table (Compare with Paper Table 11):")
        print(results_df)

# Usage
if __name__ == "__main__":
    study = FraudReplicationStudy('fraudTest.csv')
    study.run_analysis()