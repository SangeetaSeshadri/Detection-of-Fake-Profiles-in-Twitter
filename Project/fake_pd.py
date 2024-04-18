
"""
Created on Sun Mar 31 11:29:14 2024

@author: Komala Sangeeta
"""

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import  metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a Flask app
app = Flask(__name__)

def read_datasets():
    genuine_users = pd.read_csv("E://4thyear//PROJECT//proj//users.csv")
    fake_users = pd.read_csv("E://4thyear//PROJECT//proj//fusers.csv")

    x=pd.concat([fake_users,genuine_users])   
    y= len(fake_users)*[1]+len(genuine_users)*[0]
    return x,y

def preprocessing(x):          
    x.loc[:,'lang_code'] = x['lang'].map( lambda x: lang_dict[x]).astype(int)    
    x['created_at'] = pd.to_datetime(x['created_at'], utc=True).dt.strftime('%b %d %Y')
    x['created_at'] = pd.to_datetime(x['created_at'])
    current_date = pd.Timestamp.now(tz=None)
    x['account_age_days'] = (current_date - x['created_at']).dt.days

    def calculate_ffr(row):
        if row['followers_count'] != 0:
            return row['friends_count'] // row['followers_count']
        else:
            return -1

    x['ffr'] = x.apply(calculate_ffr, axis=1)
    
    
    
    feature_columns_to_use = ['statuses_count','followers_count','friends_count','favourites_count','listed_count','lang_code','ffr', 'account_age_days']
    x = x[feature_columns_to_use]
    
    return x

def extract_features_for_prediction(created_at, statuses_count, followers_count, friends_count, favourites_count, listed_count, lang):
    lang_code = lang_dict.get(lang, -1)
    created_at = pd.to_datetime(created_at, utc=True).tz_convert(None)
    account_age_days = (pd.Timestamp.now(tz=None) - created_at).days

    ffr = friends_count // followers_count if followers_count != 0 else -1
    

    feature_vector = [statuses_count, followers_count, friends_count, favourites_count, listed_count, lang_code, ffr, account_age_days]
    
    return np.array(feature_vector).reshape(1, -1) 

# Read datasets
x,y=read_datasets()
lang_list = list(enumerate(np.unique(x['lang'])))   
lang_dict = { name : i for i, name in lang_list }
x=preprocessing(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)




model =  GradientBoostingClassifier(learning_rate=0.1, n_estimators=300, max_depth=5, random_state=42)

model.fit(X_train, y_train)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    sc = int(data['statuses_count'])
    folc = int(data['followers_count'])
    fric = int(data['friends_count'])
    favc = int(data['favourites_count'])
    lc = int(data['listed_count'])
    c=data['created_at']
    l=data['lang']
    
    df = extract_features_for_prediction(c,sc,folc,fric,favc,lc,l)
    
    prediction = model.predict(df)
    
    
    # Predict on test set
    y_pred =model.predict(X_test)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    
    
    # Plot confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('static/confusion_matrixc.png')
    
    return render_template('result.html', result=prediction[0], accuracy=accuracy, precision=precision, recall=recall, f1=f1)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
