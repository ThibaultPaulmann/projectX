import sys

from models.cnn_lstm_hypernet import *

from train_outcome_model import train_outcome_model_features,prepare_features

# Train your model.
def train_challenge_model(data_folder, model_folder):
    features, metadata, outcomes, mel_spectrograms = prepare_features(data_folder, model_folder)
    rcost = 0
    rauroc = 0
    for i in range(20):
        cost, auroc = train_outcome_model_features(features, outcomes, metadata, mel_spectrograms, model_folder,ensemble_id=i)
        rcost += cost
        rauroc += auroc
    print(rcost / 20, rauroc / 20)

if __name__ == '__main__':
    # Parse the arguments.
    if not (len(sys.argv) == 3 or len(sys.argv) == 4):
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    # Define the data and model foldes.
    data_folder = sys.argv[1]
    model_folder = sys.argv[2]

    train_challenge_model(data_folder, model_folder) ### Teams: Implement this function!!!