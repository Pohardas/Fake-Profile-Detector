# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load the Data
data = pd.read_csv('fake_profiles.csv')  # Load the CSV file
print(data.head())  # Show the first few rows of the data

# Step 3: Prepare the Data
# Features (what the model will use to make predictions)
features = data.drop('is_fake', axis=1)  # Drop the 'is_fake' column
# Labels (what the model will predict)
labels = data['is_fake']

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Step 5: Train the Model
model = RandomForestClassifier()  # Use a Random Forest model
model.fit(X_train, y_train)  # Train the model on the training data

# Step 6: Test the Model
y_pred = model.predict(X_test)  # Make predictions on the test data
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy

# Step 7: Save the Model
import joblib
joblib.dump(model, 'fake_profile_detector.pkl')  # Save the model to a file
print("Model saved as 'fake_profile_detector.pkl'")