import hashlib, os, sys
sys.path.insert(0, os.path.abspath("."))
from src.train import main

DATA_PATH = "data/breast_cancer.csv"
EXPECTED_MD5 = "5afc23b9622f8f9ffa60824f4e97d0d5"

def test_data_integrity():
    assert os.path.exists(DATA_PATH), "Data file missing"
    current_md5 = hashlib.md5(open(DATA_PATH, "rb").read()).hexdigest()
    assert current_md5 == EXPECTED_MD5, f"Data drift detected! Expected {EXPECTED_MD5}, got {current_md5}"

def test_accuracy_tolerance():
    res = main(seed=42)
    assert abs(res["accuracy_mean"] - 0.97368) < 1e-3, f"Unexpected accuracy: {res['accuracy_mean']}"
