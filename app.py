from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
# def get_labels_and_texts(file):
#     labels = []
#     texts = []
#     for line in bz2.BZ2File(file):
#         x = line.decode("utf-8")
#         labels.append(int(x[9]) - 1)
#         texts.append(x[10:].strip())
#     return np.array(labels), texts
# train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')


# y=train_labels[0:500]

# #text preprocessing
# import re
# NON_ALPHANUM = re.compile(r'[\W]')
# NON_ASCII = re.compile(r'[^a-z0-1\s]')
# def normalize_texts(texts):
#     normalized_texts = []
#     for text in texts:
#         lower = text.lower()
#         no_punctuation = NON_ALPHANUM.sub(r' ', lower)
#         no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
#         normalized_texts.append(no_non_ascii)
#     return normalized_texts
        
# train_texts = normalize_texts(train_texts)

# #counter vectorize
# from sklearn.feature_extraction.text import CountVectorizer

# cv = CountVectorizer(binary=True)
# cv.fit(train_texts)
# x = cv.transform(train_texts)
# pickle.dump(cv, open('tranform.pkl', 'wb'))

# #model
# from sklearn.linear_model import LogisticRegression
# #from sklearn.metrics import accuracy_score
# #from sklearn.model_selection import train_test_split




    
# lr = LogisticRegression(C=0.1)
# clf=lr.fit(x,y)
    
# filename = 'nlp_model.pkl'
# pickle.dump(clf, open(filename, 'wb'))

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
