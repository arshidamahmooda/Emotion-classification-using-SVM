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

Performance comparison across kernels:
         accuracy  precision    recall  f1-score
linear   0.874525   0.873999  0.874525  0.873592
poly     0.644322   0.761073  0.644322  0.602744
rbf      0.856104   0.857393  0.856104  0.852399
sigmoid  0.873726   0.873040  0.873726  0.872630
✅ **Linear kernel** performed best and was selected for final training.

# Final Model Performance (Full Dataset)
Accuracy:89.9%
Macro Avg F1-Score: 0.85
Weighted Avg F1-Score:0.90
Total Tweets: 83,362 (20% test split)


