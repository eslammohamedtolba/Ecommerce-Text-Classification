from flask import Flask, render_template, request
import nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import numpy as np
import re
import joblib as jb



# Load model and vectorizer
model_path = 'PrepareModel\RandomForestClassifier.sav'
vectorizer_path = 'PrepareModel\\vectorizer.sav'
model = jb.load(model_path)
vectorizer = jb.load(vectorizer_path)

# Load stop words and stemmer
StopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Create application
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')


# Create home page route
@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

# Create predict page route
@app.route('/predict', methods = ['POST'])
def predict():
    description = request.form.get('description')
    text_cleaned = re.sub(r"[^\w\s\']+",'', description).lower()
    text_tokenized = word_tokenize(text_cleaned)
    result = []
    for word in text_tokenized:
        if word not in [string.punctuation, StopWords]:
            stemmed_word = stemmer.stem(word)
            result.append(stemmed_word)
    processed_result = ' '.join(result)
    vectorized_desc = np.mean([vectorizer.wv[word] for word in processed_result if word in vectorizer.wv], axis=0)

    prediction = model.predict([vectorized_desc])[0]
    if prediction == 0:
        prediction = 'Household'
    elif prediction == 1:
        prediction = 'Books'
    elif prediction == 2:
        prediction = 'Electronics'
    else:
        prediction = 'Clothing and Accessories'
    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)