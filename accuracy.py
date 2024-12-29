from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Split data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model on the training dataset
classifier.fit(x_train, y_train)

# Predict on the test dataset
y_pred = classifier.predict(x_test)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Generate a detailed classification report
report = classification_report(y_test, y_pred, output_dict=True)
st.write("Detailed Classification Report:")
st.json(report)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

# Split data into training and testing datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
classifier.fit(x_train, y_train)

# Predict on test data
y_pred = classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit Sidebar for Model Evaluation
st.sidebar.title("Model Evaluation")
st.sidebar.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.sidebar.write("Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
st.sidebar.json(report)

# Split data into training and testing datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
classifier.fit(x_train, y_train)

# Predict on test data
y_pred = classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit Sidebar for Model Evaluation
st.sidebar.title("Model Evaluation")
st.sidebar.write(f"Model Accuracy: {accuracy * 100:.2f}%")
st.sidebar.write("Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
st.sidebar.json(report)
