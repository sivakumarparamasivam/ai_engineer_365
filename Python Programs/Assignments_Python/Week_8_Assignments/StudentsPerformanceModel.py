# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# Step 2: Create Class StudentPerformanceModel
class StudentPerformanceModel:
    def __init__(self, filepath):
        """Initialize the StudentPerformanceModel with a data file path."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist")
        
        self.filepath = filepath
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X = self.y = None
        self.data = None

    def _check_missing_values(self):
        """Check and report missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print("\nMissing Values Found:")
            print(missing_values[missing_values > 0])
            self.data = self.data.dropna()
            print("Missing values have been removed.")

    def _check_outliers(self, columns, threshold=3):
        """Detect and handle outliers using the Z-score method."""
        for column in columns:
            z_scores = np.abs((self.data[column] - self.data[column].mean()) / self.data[column].std())
            outliers = z_scores > threshold
            if outliers.any():
                print(f"\nOutliers found in {column}: {outliers.sum()} values")
                self.data = self.data[~outliers]
                print(f"Outliers in {column} have been removed.")

    def _plot_correlations(self):
        """Plot correlation matrix of features."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()

    # Step 3: Data Preprocessing
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with validation and visualization."""
        try:
            self.data = pd.read_csv(self.filepath)
            print("Dataset Loaded Successfully ✅")
            print(f"\nDataset Shape: {self.data.shape}")
            print("\nFirst 5 Rows:\n", self.data.head())
            
            # Convert categorical variables to numeric
            for column in self.data.columns:
                if self.data[column].dtype == 'object':
                    print(f"\nConverting categorical column '{column}':")
                    # For binary Yes/No columns
                    if set(self.data[column].unique()) == {'Yes', 'No'}:
                        self.data[column] = (self.data[column] == 'Yes').astype(int)
                        print(f"Converted Yes/No to 1/0 in column '{column}'")
            
            # Data validation and cleaning
            self._check_missing_values()
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self._check_outliers(numeric_columns)
            
            # Feature correlations
            self._plot_correlations()
            
            # Identify independent and dependent variables
            self.X = self.data.drop(columns=['Performance Index'])
            self.y = self.data['Performance Index']
            
            # Scale features
            self.X = pd.DataFrame(
                self.scaler.fit_transform(self.X),
                columns=self.X.columns
            )
            
            print("\nIndependent Variables:\n", list(self.X.columns))
            print("\nDependent Variable: Performance Index")
            
        except Exception as e:
            raise Exception(f"Error in data preprocessing: {str(e)}")

    # Step 4: Train-Test Split
    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        try:
            if self.X is None or self.y is None:
                raise ValueError("Data not loaded. Call load_and_preprocess_data first.")
                
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state
            )
            print(f"\nData Split: {len(self.X_train)} Training Samples, {len(self.X_test)} Testing Samples")
            
        except Exception as e:
            raise Exception(f"Error in splitting data: {str(e)}")

    # Step 5: Model Training
    def train_model(self):
        """Train the linear regression model."""
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("Data not split. Call split_data first.")
                
            self.model.fit(self.X_train, self.y_train)
            print("\nModel Training Completed ✅")
            
            # Print model coefficients
            feature_importance = pd.DataFrame({
                'Feature': self.X.columns,
                'Coefficient': self.model.coef_
            })
            print("\nFeature Importance:")
            print(feature_importance.sort_values('Coefficient', ascending=False))
            
        except Exception as e:
            raise Exception(f"Error in model training: {str(e)}")

    # Step 6: Model Evaluation
    def evaluate_model(self):
        """Evaluate the model performance with visualizations."""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_model first.")
                
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)
            
            print("\nModel Evaluation Results:")
            print(f"Mean Squared Error (MSE): {mse:.2f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            print(f"R² Score: {r2:.2f}")
            
            # Create subplots for better visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Scatter plot of actual vs predicted values
            ax1.scatter(self.y_test, y_pred, color='blue', alpha=0.5)
            ax1.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    color='red', linewidth=2)
            ax1.set_xlabel("Actual Performance Index")
            ax1.set_ylabel("Predicted Performance Index")
            ax1.set_title("Actual vs Predicted Performance Index")
            
            # Residual plot
            residuals = self.y_test - y_pred
            ax2.scatter(y_pred, residuals, color='green', alpha=0.5)
            ax2.axhline(y=0, color='red', linestyle='--')
            ax2.set_xlabel("Predicted Performance Index")
            ax2.set_ylabel("Residuals")
            ax2.set_title("Residual Plot")
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            raise Exception(f"Error in model evaluation: {str(e)}")

    # Step 7: User Input Prediction
    def predict_user_input(self):
        """Make predictions based on user input with validation."""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train_model first.")
                
            print("\nEnter values for independent variables:")
            input_data = {}
            
            for col in self.X.columns:
                while True:
                    try:
                        val = float(input(f"Enter value for {col}: "))
                        input_data[col] = val
                        break
                    except ValueError:
                        print("Please enter a valid number.")
            
            # Convert to DataFrame to ensure proper scaling
            input_df = pd.DataFrame([input_data])
            scaled_input = self.scaler.transform(input_df)
            
            predicted_value = self.model.predict(scaled_input)[0]
            print(f"\n🎯 Predicted Performance Index: {predicted_value:.2f}")
            
            # Confidence assessment based on training data range
            if predicted_value < self.y.min() or predicted_value > self.y.max():
                print("\n⚠️ Warning: Prediction is outside the range of training data.")
                print(f"Training data range: {self.y.min():.2f} to {self.y.max():.2f}")
            
            return predicted_value
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

    def save_model_info(self, filepath):
        """Save model information and performance metrics."""
        try:
            model_info = {
                "feature_importance": dict(zip(self.X.columns, self.model.coef_)),
                "intercept": float(self.model.intercept_),
                "r2_score": r2_score(self.y_test, self.model.predict(self.X_test)),
                "rmse": float(np.sqrt(mean_squared_error(self.y_test, self.model.predict(self.X_test))))
            }
            
            with open(filepath, 'w') as f:
                import json
                json.dump(model_info, f, indent=4)
            print(f"\nModel information saved to {filepath}")
            
        except Exception as e:
            raise Exception(f"Error saving model information: {str(e)}")

# -------------------- RUN SCRIPT --------------------
if __name__ == "__main__":
    try:
        # Initialize model with dataset
        model = StudentPerformanceModel("/Users/sivakumarparamasivam/Sivakumar/python_workspace/ai_engineer_2025/Python Programs/Assignments_Python/Week_8_Assignments/Student_Performance.csv")
        
        # Execute pipeline
        print("\n1. Loading and Preprocessing Data...")
        model.load_and_preprocess_data()
        
        print("\n2. Splitting Data...")
        model.split_data()
        
        print("\n3. Training Model...")
        model.train_model()
        
        print("\n4. Evaluating Model...")
        model.evaluate_model()
        
        # Save model information
        model.save_model_info("model_info.json")
        
        # Make predictions
        while True:
            try:
                model.predict_user_input()
                
                # Ask if user wants to make another prediction
                again = input("\nWould you like to make another prediction? (yes/no): ").lower()
                if again != 'yes':
                    break
                    
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                retry = input("Would you like to try again? (yes/no): ").lower()
                if retry != 'yes':
                    break
                    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please ensure the dataset is available and properly formatted.")
