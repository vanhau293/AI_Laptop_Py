import pandas as pd
import numpy as np
import os
import joblib
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report


df = pd.read_csv('./model_laptop_text/dataset_text_train.csv') 
df = df.sample(frac=1, random_state=42)

df['Text'] = df['Text'].str.lower().str.replace('[^\w\s]', '', regex=True)

X = df['Text'].apply(lambda x: x.split())
y = df['Demand']

model = Word2Vec(sentences=X, vector_size=100, window=5, min_count=1, sg=0)
# model.wv.save("dataset_text_w2v.model")

def text_to_vector(text):
    words = text
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return None
    return sum(word_vectors) / len(word_vectors)

X = X.apply(text_to_vector)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC()
clf.fit(X_train.tolist(), y_train)

y_pred = clf.predict(X_test.tolist())
print(classification_report(y_test, y_pred))


joblib.dump(clf, './model_laptop_text/svm_model_text.pkl')
model.save("./model_laptop_text/dataset_text_w2v.bin")

# loaded_svm_model = joblib.load('svm_model_text.pkl')

df_test = pd.read_csv('./model_laptop_text/dataset_text_test.csv') 

_x_test=df_test['Text']
_y_test=df_test['Demand']

# input_text='Tôi thích laptop mỏng nhẹ'

# input_text = input_text.lower().replace('[^\w\s]', '') 
 
def text_to_vector_pred(text):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return None
    return sum(word_vectors) / len(word_vectors)

def predict_laptop(input_text):
    input_vector = text_to_vector_pred(input_text)
    loaded_svm_model = joblib.load('./model_laptop_text/svm_model_text.pkl')

    if input_vector is not None:
        predicted_label = loaded_svm_model.predict([input_vector.tolist()])[0]
        # print(predicted_label)
        return predicted_label
    else:
        print("Error!")

_y_pred=[]
for text in _x_test:
    input_text = text.lower().replace('[^\w\s]', '') 
    pred= predict_laptop(input_text)
    _y_pred.append(pred)

print("Predict:",_y_pred)
print("Truth:  ",list(_y_test))
print(predict_laptop("Tôi muốn mua laptop chơi game"))


