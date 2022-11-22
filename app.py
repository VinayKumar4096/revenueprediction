#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
from flask import jsonify

import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler


# In[3]:


app = Flask(__name__)
model = pickle.load(open('lgbm_model.pkl', 'rb'))


# In[4]:


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


# In[7]:


@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        fullVisitorId = float(request.form['fullVisitorId'])
        prediction=model.predict([[fullVisitorId]])
        output=round(prediction[0],2)
        return render_template('index.html',prediction_text="Revenue Prediction is {}".format(output))
    else:
        return render_template('index.html')
if __name__=="__main__":
    
    app.run(debug=True)


# In[ ]:




