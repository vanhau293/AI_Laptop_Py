import joblib
from gensim.models import Word2Vec


# input_text='Tôi thích laptop mỏng nhẹ'
# input_text=input("Input text: ")

w2v_model= './model_laptop_text/model_laptop_text/dataset_text_w2v.bin'
svm_model='./model_laptop_text/model_laptop_text/svm_model_text.pkl'

model_w2v=Word2Vec.load(w2v_model)
loaded_svm_model = joblib.load(svm_model)



def text_to_vector_pred(text):
    words = text.split()
    word_vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
    if not word_vectors:
        return None
    return sum(word_vectors) / len(word_vectors)

def predict_demand(input_text):
    input_text = input_text.lower().replace('[^\w\s]', '') 
    input_vector = text_to_vector_pred(input_text)

    if input_vector is not None:
        predicted_label = loaded_svm_model.predict([input_vector.tolist()])[0]
        return predicted_label
    else:
        return "Error!"

# predict_demand(input_text) # 0 1 2
