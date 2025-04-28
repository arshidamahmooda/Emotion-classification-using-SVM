

**Emotion Classification Using LinearSVC** project:

---

# Emotion Classification Using LinearSVC

This project builds an **Emotion Classifier** using **TF-IDF vectorization** and a **Linear Support Vector Machine (LinearSVC)** model. It predicts emotions from input text based on supervised learning.

---

## Project Overview

The key steps in the project are:
- Load and preprocess a text dataset labeled with emotions.
- Transform the text into numerical features using **TF-IDF**.
- Train a **LinearSVC** classifier.
- Evaluate the model using a **classification report**.

---

## Requirements

You need the following Python libraries installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

Or you can install them all at once:

```bash
pip install -r requirements.txt
```

> **Note:** If you want, I can also create a `requirements.txt` for you.

---

## Dataset

- The dataset used is `emotions.csv`.
- It contains two columns:
  - `text`: the textual data (e.g., sentences expressing emotions).
  - `label`: the emotion category (e.g., "happy", "sad", "angry", etc.).

Example of the data:

| text                      | label  |
|----------------------------|--------|
| "I feel great today!"      | happy  |
| "This is so frustrating."  | angry  |

---

## File Descriptions

### `emotion_classifier.py`
Contains the following steps:
- Load the dataset.
- Handle missing labels.
- Text preprocessing with **TF-IDF Vectorizer**.
- Split into training and testing datasets.
- Train the **LinearSVC** model.
- Predict and evaluate using **classification_report**.

---

## How to Run

1. Make sure your `emotions.csv` file path is correctly specified.
2. Run the script:

```python
python emotion_classifier.py
```

3. The script will:
   - Train the model.
   - Output a **classification report** showing precision, recall, and F1-score.

Example output:

```
              precision    recall  f1-score   support

        angry       0.92      0.89      0.90       100
         fear       0.85      0.88      0.86       100
         ...
    accuracy                           0.88       600
   macro avg       0.88      0.88      0.88       600
weighted avg       0.88      0.88      0.88       600
```

---

## Notes

- **TF-IDF** converts text into numeric vectors based on word importance.
- **LinearSVC** is chosen for its efficiency on high-dimensional, sparse data like text.
- Missing labels are handled by dropping them.
- You can improve results by tuning hyperparameters (e.g., adjusting `C` value for SVM).

---

## License

This project is licensed under the **MIT License**.
