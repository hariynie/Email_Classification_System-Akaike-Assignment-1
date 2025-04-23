Email Classification System – Akaike Assignment 1
This project is a machine learning-based email classification system developed as part of Akaike's Assignment 1. It aims to categorize emails into appropriate classes using a trained model.​

📁 Project Structure
Akaike_Assignment_Email_Classification (1).pdf: Assignment brief detailing the project requirements.

app-checkpoint.py: Python script for deploying the classification model.

Training-checkpoint.ipynb: Jupyter notebook for training the model.

TestingApp-checkpoint.ipynb: Notebook for testing the trained model.

home.html: HTML template for the application's homepage.

result.html: HTML template to display classification results.

input.xlsx: Sample input data for testing.

test_cnn.pkl, test_label.pkl, tokenizer.pkl, train_word_index.pkl: Serialized objects related to the trained model and tokenizer.​

🚀 Getting Started
Prerequisites
Python 3.x

Jupyter Notebook

Required Python libraries (e.g., TensorFlow, Keras, pandas, scikit-learn)​

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/hariynie/Email_Classification_System-Akaike-Assignment-1.git
Navigate to the project directory:

bash
Copy
Edit
cd Email_Classification_System-Akaike-Assignment-1
Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Note: Ensure that a requirements.txt file is present with all necessary dependencies listed.​

🧠 Model Training
Open Training-checkpoint.ipynb in Jupyter Notebook to train the model on the provided dataset. This notebook includes data preprocessing, model architecture, training, and evaluation steps.​

🧪 Model Testing
Use TestingApp-checkpoint.ipynb to test the trained model's performance on new data. This notebook demonstrates loading the model and making predictions.​

🌐 Web Application
The app-checkpoint.py script sets up a web application using Flask (assumed based on common practices) to interact with the email classification model. The home.html and result.html templates render the user interface.​

📊 Sample Data
The input.xlsx file contains sample emails for testing the classification system.​

📦 Serialized Objects
test_cnn.pkl: Trained CNN model.

test_label.pkl: Label encoder for the classification labels.

tokenizer.pkl: Tokenizer used for text preprocessing.

train_word_index.pkl: Word index mapping from the training data.​

