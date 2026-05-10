import yaml
import pickle
import os

def read_yaml(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def save_object(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)

def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)