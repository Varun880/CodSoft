# Movie Genre Classification using ML & Movie Title & Script Generator using Generative AI

This project focuses on classifying **movie genres** using machine learning based on **titles and descriptions**. It also incorporates Generative AI to create synthetic movie ideas, showcasing how generative models can complement traditional ML pipelines.

## Project Structure

├── GenreClassifier_EDA.ipynb # Data conversion, cleaning, and EDA
├── GenreClassifier.ipynb # ML training with TF-IDF + SVM, Logistic Regression, Naive Bayes
├── GenreGenerator.ipynb # Movie title and plot generation using LLaMA 3.2
├── Datasets/
│ ├── train_data.txt
│ ├── test_data.txt
│ ├── test_data_solution.txt
│ ├── train_data.csv
│ ├── test_data.csv
│ └── test_data_solution.csv
├── models/
│ ├── svm_genre_prediction_model.pkl
│ ├── tfidf_vectorizer.pkl
│ └── label_encoder.pkl

###  `GenreClassifier_EDA.ipynb`
- Converts raw `.txt` data to `.csv`.
- Performs exploratory data analysis (EDA), including:
  - Genre distribution
  - Null value checks
  - Unique genre counts

### `GenreClassifier.ipynb`
- Preprocessing: combines title and description, removes stopwords and punctuation.
- Feature extraction: uses **TF-IDF** with trigrams (1,2,3) and 20,000 max features.
- Trains and evaluates:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Multinomial Naive Bayes

### `GenreGenerator.ipynb`
- Uses **LLaMA 3.2 (via Ollama)** to generate creative movie titles and plot summaries.
- User inputs a genre; model responds with a synthetic movie concept.

## Best Model Summary

- **Algorithm**: Support Vector Machine (SVM)
- **Vectorizer**: TF-IDF with n-grams (1,2,3), 20k features
- **Hyperparameter**: `C = 0.1`
- **Test Accuracy**: **60.03%**
- **Saved at**: `models/svm_genre_prediction_model.pkl`
