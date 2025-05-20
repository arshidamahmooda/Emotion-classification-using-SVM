#  Emotion Classification Using LinearSVM

This project builds an Emotion Classifier using **TF-IDF vectorization** and a **Linear Support Vector Machine (LinearSVC)** model. It classifies English-language tweets into **six basic emotions** using a pre-labeled dataset of over 416,000 tweets.

##  Folder Structure

emotion-svm/
├── data/
│   └── emotions.csv             # Input dataset (416k tweets)
├── models/
│   ├── svm_model_linear.pkl     # Trained LinearSVC model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
├── notebooks/
│   └── emotion_classifier.ipynb # Jupyter notebook (optional)
├── src/
│   └── train_model.py           # Main training pipeline
├── results/
│   └── confusion_matrix.png     # Sample confusion matrix
├── README.md
└── requirements.txt

##  Project Overview

* Supervised learning on tweet text
* Text preprocessing (cleaning, stopword removal)
* Feature extraction using **TF-IDF**
* Emotion classification with **SVM (Linear, RBF, Poly kernels)**
* Model comparison, confusion matrix visualization
* Final model saved as `.pkl` files for deployment or inference

## 🧪 Kernel Performance Comparison (on 5K Sample)

| Kernel | Accuracy  | Training Time |
| ------ | --------- | ------------- |
| Linear | **83.2%** | 0.23 sec      |
| RBF    | 74.5%     | 2.92 sec      |
| Poly   | 59.9%     | 9.52 sec      |

✅ **Linear kernel** performed best and was selected for final training.

# Final Model Performance (Full Dataset)
Accuracy:89.9%
Macro Avg F1-Score: 0.85
Weighted Avg F1-Score:0.90
Total Tweets: 83,362 (20% test split)


