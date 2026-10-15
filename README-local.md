Cynaris Internship Project — RetailSense Sentiment Analysis

Project Overview
This project develops a machine learning pipeline to classify e-commerce product reviews into positive, negative, and neutral sentiment categories.

Technologies Used
1.Python
2.Pandas and NumPy
3.Scikit-learn
4.TF-IDF Vectorization
5.Logistic Regression, Decision Tree, and Random Forest
6.Matplotlib and Seaborn

Project Workflow
1.Load and inspect the product review dataset.
2.Clean and preprocess the data.
3.Convert review text into numerical features using TF-IDF.
4.Train and compare machine learning models.
5.Evaluate the selected model using accuracy, precision, recall, F1-score, and a confusion matrix.
6.Save the trained model and evaluation results.

Selected Model
Logistic Regression
Evaluation Results
Accuracy: 96.62%
Macro Precision: 0.9700
Macro Recall: 0.7917
Macro F1-score: 0.8331

Project Files
main.py — Main model training and evaluation script.
DataCheckWeek2.py — Dataset inspection and checking.
aiml_product_reviews_corrected.csv — Corrected review dataset.
outputs/ — Generated outputs and evaluation artifacts, if saved there.

Note
These results are from the current held-out test split. Neutral sentiment recall needs improvement.
