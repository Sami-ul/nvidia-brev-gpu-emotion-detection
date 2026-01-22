import kagglehub
from dotenv import set_key

path = kagglehub.dataset_download("ananthu017/emotion-detection-fer")

set_key(dotenv_path="src/.env", key_to_set="DATASET_PATH", value_to_set=path)
print("Set DATASET_PATH env variable to: ", path)
