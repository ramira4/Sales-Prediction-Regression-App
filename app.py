import numpy as np
from flask import Flask, request, render_template
import pickle
import os
from data import *
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
# build_figs()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = np.round(prediction[0], 2)
    print(output)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return e
    return render_template("index.html", error_text='Error: Please try again')


if __name__ == "__main__":
    app.run(debug=True)