import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the font dataset
try:
    with open("google_fonts_extended_dataset_with_category.json", "r") as f:
        font_data = json.load(f)
        print(f"Loaded dataset with {len(font_data)} entries.")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# Convert data to DataFrame
df = pd.DataFrame(font_data)
print("Dataset columns:", df.columns)

# Define the features for analysis
features = ["weight", "italic", "width", "line_height", "x_height", "contrast", "curvature", "category"]

# Check that all required columns are present in the dataset
missing_columns = [feature for feature in features if feature not in df.columns]
if missing_columns:
    print(f"Missing columns in dataset: {missing_columns}")
    exit()
else:
    print("All required columns are present.")

# Preprocess categorical features using LabelEncoder
label_encoders = {}
for feature in features:
    df[feature] = df[feature].astype(str)
    if df[feature].dtype == "object":
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].fillna("Unknown"))
        label_encoders[feature] = le

# Scale features for uniformity
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# Function to suggest the most contrasting fonts based on a target font
def suggest_most_contrasting_fonts(target_font, n_suggestions=5):
    # Find the target font in the dataset
    target_data = df[df["family"] == target_font]
    if target_data.empty:
        print(f"Font '{target_font}' not found in dataset.")
        return None

    print(f"Target font '{target_font}' found. Proceeding with contrasting suggestions...")

    # Get the target font's features and scale them
    target_features = target_data[features].iloc[0].values.reshape(1, -1)
    target_scaled = scaler.transform(target_features)

    # Calculate distances to all fonts and sort by descending distance for contrast
    distances = np.linalg.norm(X - target_scaled, axis=1)
    contrasting_indices = np.argsort(distances)[::-1]  # Sort by farthest distances

    # Select top 'n_suggestions' contrasting fonts
    contrasting_fonts = df.iloc[contrasting_indices]["family"].head(n_suggestions).values

    print(f"Most contrasting fonts for '{target_font}': {contrasting_fonts}")
    return contrasting_fonts

# Example usage
target_font = "Roboto"  # Replace with a font from your dataset
contrasting_fonts = suggest_most_contrasting_fonts(target_font)
if contrasting_fonts is not None:
    print(f"Most contrasting fonts for '{target_font}': {contrasting_fonts}")
else:
    print("No contrasting fonts found.")
