import pickle

def load_processed_dataset(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
