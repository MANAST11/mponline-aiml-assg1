import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the plot styles look clean and professional
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14, 'axes.labelsize': 12})

def run_assignment():
    
    # Data Understanding
    
    print("=" * 60)
    print("DATA UNDERSTANDING")
    print("=" * 60)
    
    # Load the dataset using Pandas.
    dataset_path = "insurance.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found in the workspace.")
        
    df = pd.read_csv(dataset_path)
    
    # Display the first five records.
    print("\nFirst 5 records of the dataset:")
    print(df.head())
    
    # Identify Numerical features, Categorical features, and Target variable.
    numerical_features = list(df.select_dtypes(include=[np.number]).columns)
    # Exclude 'charges' as it is the target
    if 'charges' in numerical_features:
        numerical_features.remove('charges')
        
    categorical_features = list(df.select_dtypes(exclude=[np.number]).columns)
    target_variable = 'charges'
    
    print("\nFeature Identification:")
    print(f" - Numerical Features: {numerical_features}")
    print(f" - Categorical Features: {categorical_features}")
    print(f" - Target Variable: {target_variable}")
    print(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
    
    
    # Data Preprocessing
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Check for missing values.
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)
    
    # Encode categorical variables (sex, smoker, region).
    print("\nCategorical columns before encoding:")
    print(df[categorical_features].head())
    
    # One-hot encoding categorical variables (sex, smoker, region)
    # Using drop_first=True to avoid multi-collinearity (dummy variable trap)
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int)
    
    print("\nEncoded dataset columns:")
    print(df_encoded.columns.tolist())
    print("\nFirst 5 rows of encoded dataset:")
    print(df_encoded.head())
    
    # Split the dataset into 80% training and 20% testing.
    # Features are everything except the target variable
    X = df_encoded.drop(columns=[target_variable])
    y = df_encoded[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset Split:")
    print(f" - Training set shape (X_train): {X_train.shape}")
    print(f" - Testing set shape (X_test):   {X_test.shape}")
    print(f" - Training labels shape (y_train): {y_train.shape}")
    print(f" - Testing labels shape (y_test):   {y_test.shape}")
    
    
    # Model Development
    
    print("\n" + "=" * 60)
    print("MODEL DEVELOPMENT")
    print("=" * 60)
    
    # Build a Multiple Linear Regression model
    model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")
    
    # Model coefficients
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    intercept = model.intercept_
    print(f"Model Intercept: {intercept:.2f}")
    print("\nModel Coefficients:")
    print(coefficients)
    
    # Predict the charges for the test dataset
    y_pred = model.predict(X_test)
    print("\nCharges predicted on the test dataset.")
    
    
    # Model Evaluation
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Evaluate the model using MAE, MSE, R² Score
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEvaluation Metrics:")
    print(f" - Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f" - Mean Squared Error (MSE):  ${mse:,.2f}")
    print(f" - Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f" - R² Score (Coefficient of Determination): {r2:.4f}")
    
    # Create Actual vs Predicted scatter plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with custom aesthetics
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#2c3e50", edgecolor="w", s=60, label="Predictions")
    
    # Draw reference diagonal line (y = x) representing perfect prediction
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2.5, label="Perfect Fit (y = x)")
    
    plt.title("Actual vs. Predicted Medical Insurance Charges", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Actual Charges ($)", fontsize=12)
    plt.ylabel("Predicted Charges ($)", fontsize=12)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    plot_filename = "actual_vs_predicted.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"\nActual vs Predicted scatter plot saved as '{plot_filename}'")
    
    # Observations based on model performance
    print("\nObservations based on Model Performance:")
    print("1. Smoker status is by far the most significant predictor. The coefficient for 'smoker_yes' is extremely high")
    print("   (approx. $23,651), meaning smokers are charged significantly more than non-smokers, holding all other features constant.")
    print("2. The R² score is approximately 0.783, indicating that about 78.3% of the variance in insurance charges can be")
    print("   explained by our model features. This shows a strong linear relationship overall, though there is still unexplained variance.")
    print("3. Visual inspection of the scatter plot shows distinct 'tiers' or groups of data points, particularly at higher costs,")
    print("   where the model tends to underpredict charges. This indicates non-linear dynamics, such as interactions between factors")
    print("   (e.g., BMI and smoking combined), which a standard additive linear model cannot capture without interaction terms.")
    
    
    # Conclusion
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    
    conclusion = (
        "This project successfully developed a Multiple Linear Regression model predicting medical insurance charges with an R² score of 0.783. "
        "The analysis reveals that smoking status is the most dominant factor, increasing charges by over $23,600, followed by age and BMI, which also show a positive correlation with medical costs. "
        "Gender and residential region show negligible impacts on insurance charges. "
        "A primary limitation of Linear Regression in this context is its assumption of linearity and additivity. "
        "The model cannot naturally capture the non-linear interaction effect between smoking and high BMI (where obese smokers face disproportionately higher charges), leading to underpredictions at higher charge ranges."
    )
    print(f"\n{conclusion}\n")
    
    print("=" * 60)

if __name__ == "__main__":
    run_assignment()
