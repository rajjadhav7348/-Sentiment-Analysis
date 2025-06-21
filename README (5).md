
# Sentiment Analysis using NLP

This project performs sentiment analysis on a dataset of customer reviews using TF-IDF vectorization and Logistic Regression.

## 📌 Objective

To classify customer reviews as Positive, Negative (and optionally Neutral) using natural language processing and machine learning.

## 🧰 Tools & Libraries Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib / Seaborn (for visualization)

## ⚙️ Project Steps

1. **Import Libraries**
2. **Load and Preprocess Data**
3. **Text Cleaning using NLTK**
4. **Vectorization using TF-IDF**
5. **Model Building with Logistic Regression**
6. **Model Evaluation (Accuracy, Confusion Matrix)**
7. **Model Saving (Optional)**

## 📂 Dataset

Dataset should contain at least two columns:
- `review`: The text content of the review
- `sentiment`: The target label (e.g., Positive, Negative)

> A sample dataset is included for testing if none is provided.

## 🧪 How to Run

1. Clone this repository or download the notebook.
2. Ensure required libraries are installed:
    ```bash
    pip install pandas scikit-learn nltk matplotlib seaborn
    ```
3. Run the Jupyter Notebook step-by-step.
4. Optionally upload your own `customer_reviews.csv` file.

## 📊 Example Output

- Accuracy Score
- Confusion Matrix
- Classification Report

## 📁 Output Files

- `logistic_model.pkl` — Trained model
- `tfidf_vectorizer.pkl` — TF-IDF vectorizer

## ✍️ Author

Rajnandan Jadhav  
GitHub: [rajjadhav7348](https://github.com/rajjadhav7348)
