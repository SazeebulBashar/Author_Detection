import numpy as numpy
import pickle as pkl
from flask import Flask, request,jsonify,json,render_template,Markup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


# model = pkl.load(open('x_model.pkl','rb'))
model = pkl.load(open('SVM_model.pkl','rb'))
cv = pkl.load(open('cv.pkl','rb'))
le = pkl.load(open('le.pkl','rb'))

model_poem  = pkl.load(open('x_model_for_PoemStory.pkl','rb'))
cv_poem = pkl.load(open('cv_poem.pkl','rb'))
le_poem = pkl.load(open('le_poem.pkl','rb'))


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

    x2 = cv_poem.transform([sentence]).toarray()
    lang2 = model_poem.predict(x2) 
    lang2 = le_poem.inverse_transform(lang2)

    result= checkAuthor(lang[0]) +"'s  " + checkGenre(lang2[0])
    return render_template('index.html',res = result)


def checkAuthor(value):
    if value == 1:
        return 'Rabindranath Tagore'
    if value == 2:
        return 'Bankimchandra Chattapaddhaya'
    if value == 3:
        return 'Jasim Uddin'
    return 'Kazi Nazrul Islam'

def checkGenre(value):
    if value == 1:
        return "Poem"
    return "Novel/Story"

if __name__ == '__main__':
    app.run(debug = True)