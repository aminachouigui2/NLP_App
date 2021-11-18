#!/usr/bin/env python
# coding: utf-8
import streamlit as st

#for data visualization
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
#stop_words = set(stopwords.words("german"))
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

# Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#to save the model
import joblib
import pickle

#for Metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

#for Arrays
import numpy as np


st.title("NLP (text classification) APP")
st.write('The dataset used for the classifiers training/test, was downloaded from https://tblock.github.io/10kGNAD/')

#for lemmatization
class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, articles):
		return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


if st.button('Investigate the dataset'):
	#st.markdown('##### Investigate the dataset')
	df = pd.read_csv('10kGNAD.csv', sep=';', header=None, usecols=[0,1], names=['category', 'text']) 	
	sns.set_style("whitegrid")
	x=df['category'].value_counts()
	#print(x)
	sns.barplot(x.index,x).set(title='Dataset visualization', 	xlabel='Categories', ylabel='Number of articles')
	plt.xticks(rotation=40)
	x.columns=["Category", "Number of Articles"]
	st.write('Number of articles in each category', x)
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()

	

st.markdown('##### Enter a text (in german) to predit its class')

txt = st.text_area(label='', value='', placeholder= 'Enter a text (in german)')

if txt:
	#load the model from disk
	model = open("clModel.pkl","rb")
	loaded_model = joblib.load(model)
	#load the vectorizer from disk
	vec = open("tfidfVec.pkl","rb")
	loaded_vec = joblib.load(vec)
	#predict the class for the given text
	Y_pred_class = loaded_model.predict(loaded_vec.transform([txt]))
	st.write('the predicted class is:', Y_pred_class[0])


