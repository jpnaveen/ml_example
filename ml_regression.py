import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
# Ensure Housing.csv is in the same directory as this script
try:
    # dataset is taken from kagel
    #https://www.kaggle.com/datasets/yasserh/housing-prices-dataset?resource=download&select=Housing.csv
    df = pd.read_csv('dataset/Housing.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Housing.csv not found. Please download it from Kaggle.")
    exit()

# 2. Preprocessing
# Identify categorical columns that need encoding
categorical_cols = [
    'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
    'airconditioning', 'prefarea', 'furnishingstatus'
]

# Use LabelEncoder for binary/categorical features
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Define Features (X) and Target (y)
X = df.drop('price', axis=1)
y = df['price']

# 4. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train the Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"Root Mean Squared Error: {rmse:,.2f}")
print(f"R-squared Score: {r2:.4f}")

# 8. Showcase Coefficients
print("\n--- Feature Coefficients ---")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients.sort_values(by='Coefficient', ascending=False))

# Example: Testing with a single hypothetical data point
# Using the first row of the test set as an example
sample_house = X_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample_house)
print(f"\nPredicted price for sample: {prediction[0]:,.2f}")
print(f"Actual price for sample: {y_test.iloc[0]:,.2f}")

