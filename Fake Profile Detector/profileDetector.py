# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 2: Load and Combine Data
real_profiles = pd.read_csv('real_profiles.csv')  # Load real profiles
fake_profiles = pd.read_csv('fake_profiles.csv')  # Load fake profiles

# Add a label column to distinguish real and fake profiles
real_profiles['is_fake'] = 0  # 0 for real profiles
fake_profiles['is_fake'] = 1  # 1 for fake profiles

# Combine the two datasets into one
data = pd.concat([real_profiles, fake_profiles], ignore_index=True)

# Step 3: Preprocess Data
data = data.fillna(0)  # Fill missing values with 0

# Step 4: Prepare the Data for Training
features = data.drop('is_fake', axis=1)  # Drop the target column
labels = data['is_fake']  # Keep only the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestClassifier()  # Use a Random Forest model
model.fit(X_train, y_train)  # Train the model on the training data

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)  # Make predictions on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy

# Step 7: Save the Model
joblib.dump(model, 'fake_profile_detector.pkl')  # Save the model to a file
print("Model saved as 'fake_profile_detector.pkl'")