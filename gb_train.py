import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# =====================
# Load dataset
# =====================
df = pd.read_csv("laptop_price.csv")
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")

# =====================
# Drop unnecessary columns
# =====================
columns_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'name', 'processor', 'CPU', 'GPU']

for col in columns_to_drop:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print(f"Columns dropped. New shape: {df.shape}")

# =====================
# Clean Ram column
# =====================
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)

# =====================
# Clean ROM column
# =====================
def clean_rom(rom_value):
    if 'TB' in rom_value:
        return int(rom_value.replace('TB', '')) * 1024
    else:
        return int(rom_value.replace('GB', ''))

df['ROM'] = df['ROM'].apply(clean_rom)

print("Data cleaning completed!")

# =====================
# Target and features
# =====================
X = df.drop('price', axis=1)
y = df['price']

# =====================
# Column split
# =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# =====================
# Preprocessing
# =====================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

# =====================
# Gradient Boosting Model (Tuned Parameters)
# =====================
gb_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    n_estimators=188,
    random_state=42
)

# =====================
# Full Pipeline
# =====================
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', gb_model)
])

# =====================
# Train-test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# =====================
# Train model
# =====================
print("\nTraining Gradient Boosting model...")
gb_pipeline.fit(X_train, y_train)

# =====================
# Evaluation
# =====================
y_pred = gb_pipeline.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE:")
print("="*60)
print(f"R² Score: {r2:.4f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"MAE: ₹{mae:.2f}")
print("="*60)

# =====================
# Save model (IMPORTANT)
# =====================
with open("laptop_gb_pipeline.pkl", "wb") as f:
    pickle.dump(gb_pipeline, f)

print("\nGradient Boosting pipeline saved as laptop_gb_pipeline.pkl")
print("Training completed successfully!")