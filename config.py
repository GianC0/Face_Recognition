"""Configuration file for the project

If imported, it contains only one variable named config. config is a dictionary
containing other dictionaries to better differentiate the different
configurations. An example of importing it would be:

from config import config as settings

If used as a standalone program, it prints the current values for all of the
settings
"""

config = {
    "dirs": {
        "train": "train_images",
        "test": "test_images",
        "real": "real_images",
    },
    "training": {
        "valid_size": 0.1,
        "batch_size": 32,
        "n_epochs": 10,
        "learning_rate": 0.01,
        "min_valid_loss": 1,
    },
    "usage": {
        "input": "real_images",
        "output": "real_images",
    },
    "pyramid": {
        "iou_threshold": 0.1,
        "prob_threshold": 0.9999,
    }
}

if __name__ == "__main__":
    print("Current settings:")
    for key_1 in config:
        print(f"-> {key_1}")
        for key_2 in config[key_1]:
            print(f"\t-> {key_2}: {config[key_1][key_2]}")