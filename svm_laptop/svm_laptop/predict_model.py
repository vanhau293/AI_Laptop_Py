import joblib
import numpy as np
import pandas as pd
# ['Price', 'RAM','Hard Drive', 'Weight','Brand Rank','CPU Rank','GPU Rank','Inch']


def predict_laptop(input_laptop):
    data_ip = np.array(input_laptop)
    data_reshaped = data_ip.reshape(1, -1)

    loaded_svm_model = joblib.load("./svm_laptop/svm_laptop/svm_model.pkl")

    pred= loaded_svm_model.predict(data_reshaped)
    return pred[0] # 0 1 2 
