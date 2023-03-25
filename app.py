import numpy as numpy
import pickle as pkl
from flask import Flask, request,jsonify,json,render_template,Markup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


# model = pkl.load(open('x_model.pkl','rb'))
model = pkl.load(open('SVM_model.pkl','rb'))
cv = pkl.load(open('cv.pkl','rb'))
le = pkl.load(open('le.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    sentence = request.form.get('sentence')
    x = cv.transform([sentence]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang) 

    return render_template('index.html',result=lang[0])

if __name__ == '__main__':
    app.run(debug = True)