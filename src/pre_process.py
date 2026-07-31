import os
import yaml
import pandas as pd


def load_params():
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["pre_process"]


def preprocess(input_path, output_path):
    # Data load karein
    df = pd.read_csv(input_path)
    
    # Processed folder ensure karein
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Processed output save karein
    df.to_csv(output_path, index=False)
    print(f"[INFO] Preprocessed data saved to: {output_path}")


if __name__ == "__main__":
    params = load_params()
    preprocess(params["input"], params["output"])
