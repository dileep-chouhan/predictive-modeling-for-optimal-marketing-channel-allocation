import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_campaigns = 100
data = {
    'Channel': np.random.choice(['Online', 'TV', 'Radio', 'Print'], size=num_campaigns),
    'Spend': np.random.uniform(1000, 10000, size=num_campaigns),
    'Acquisitions': np.random.randint(10, 500, size=num_campaigns),
    'ROI': np.random.uniform(0.5, 5, size=num_campaigns) # ROI is a random number between 0.5 and 5
}
df = pd.DataFrame(data)
# Introduce some correlation between spend and acquisitions for a more realistic dataset.
df['Acquisitions'] = df['Spend'] * 0.01 + np.random.normal(0, 50, num_campaigns)
df['Acquisitions'] = df['Acquisitions'].clip(lower=0) # Ensure no negative acquisitions
# --- 2. Data Cleaning and Preprocessing ---
# One-hot encode the categorical variable 'Channel'
df = pd.get_dummies(df, columns=['Channel'], drop_first=True)
# --- 3. Predictive Modeling ---
# Define features (X) and target (y)
X = df.drop('ROI', axis=1)
y = df['ROI']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 4. Visualization ---
# Visualize the relationship between spend and ROI
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Spend', y='ROI', data=df, hue='Channel')
plt.title('Spend vs. ROI')
plt.xlabel('Spend')
plt.ylabel('ROI')
plt.savefig('spend_vs_roi.png')
print("Plot saved to spend_vs_roi.png")
# Visualize feature importances (if applicable, for more complex models)
# This part is commented out because it's not directly applicable to linear regression,
# but it's good practice to include such analysis for more sophisticated models.
# feature_importances = pd.Series(model.feature_importances_, index=X.columns)
# feature_importances.plot(kind='barh')
# plt.title('Feature Importances')
# plt.savefig('feature_importances.png')
# print("Plot saved to feature_importances.png")