import pickle

# Specify the path to your PKL file
pkl_file_path = './svm_laptop/svm_model.pkl'

# Load the PKL file
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# Display the contents
print(data)