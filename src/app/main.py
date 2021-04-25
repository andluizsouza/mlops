from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import os
import pickle

app = Flask(__name__)
# Add authentication
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')
basic_auth = BasicAuth(app)


# main endpoint
@app.route('/')
def home():
    return 'My first API'


# new endpoint
@app.route('/sentiment/<phrase>')
# the input is the final part of url
@basic_auth.required
def sentiment(phrase):
    tb = TextBlob(phrase)
    tb_en = tb.translate(to='en')
    polar = tb_en.sentiment.polarity
    return 'polarity is {}'.format(polar)


cols = ['tamanho', 'ano', 'garagem']
model = pickle.load(open('../../models/model.sav', 'rb'))


@app.route('/request_post/', methods=['POST'])
# the input from the method POST: use postman
@basic_auth.required
def request_post():
    user_input = request.get_json()
    x = [user_input[col] for col in cols]
    y = model.predict([x])
    return jsonify(price=y[0])


# specify  the host
app.run(debug=True, host='0.0.0.0')
