# Ecommerce Text Classification
The "Ecommerce Text Classification" project aims to classify product descriptions from an e-commerce dataset into predefined categories. 
The dataset consists of product descriptions categorized into four distinct classes: Household, Books, Electronics, and Clothing_Accessories. 
The project employs various machine learning models to classify these descriptions and uses a Flask web application for user-friendly deployment.

![Image about the final project](<Ecommerce Text Classification.png>)

## Prerequisites

- Python 3.x
- Flask
- NLTK
- NumPy
- Joblib
- Scikit-learn
- XGBoost
- Gensim
- Pandas
- Matplotlib
- Seaborn

## Overview of the code

1. **Data Preprocessing**
   - Load and clean the dataset.
   - Handle missing values and standardize category names.
   - Perform text preprocessing including tokenization, stemming, and removal of stop words.

2. **Exploratory Data Analysis (EDA)**
   - Visualize category distribution using bar plots and pie charts.
   - Generate word clouds for different categories to understand common words and patterns.

3. **Handling Class Imbalance**
   - Balance the dataset by oversampling the minority class.
   - Shuffle and reset index to ensure data integrity.

4. **Feature Extraction**
   - Use Word2Vec for text vectorization.
   - Convert text data into numerical features suitable for machine learning models.

5. **Model Training and Evaluation**
   - Train and evaluate multiple classifiers including Logistic Regression, SVM, XGBoost, Decision Tree, Random Forest, and Gradient Boosting.
   - Evaluate models using accuracy, confusion matrix, and classification report.

6. **Deployment**
   - Develop a Flask web application to deploy the trained model.
   - Implement text preprocessing and prediction logic in the Flask app.


## Benefits and Key Techniques

- **Improved Classification Accuracy**: Using a variety of classifiers and techniques such as Word2Vec and ensemble methods enhances classification performance.
- **Class Imbalance Handling**: Techniques like oversampling ensure that the model performs well across all categories.
- **Interactive Web Interface**: Flask deployment provides a user-friendly interface for real-time text classification.


## Deployment with Flask

1. **Setup**
   - Install required packages: Flask, NLTK, NumPy, Joblib.
   - Load the trained model and vectorizer.
   - Create routes for home and prediction pages.

2. **Running the Application**
   - Start the Flask server using `app.run(debug=True)`.
   - Access the application via a web browser.
  

## Accuracy

- **Random Forest Classifier** achieved the highest accuracy of 87% among the tested models.
- All models were evaluated for accuracy and other metrics such as precision, recall, and F1-score.
- Results are displayed in the confusion matrix and classification report.


## Contributions

Contributions to this project are welcome! Feel free to submit issues, enhancements, or feature requests. 
Fork the repository and create a pull request for any improvements or bug fixes.

