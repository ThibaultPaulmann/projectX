import tqdm
from helper_code import *
import numpy as np, os
import torch
import torchaudio.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.cnn_lstm_hypernet import *
from torch.optim.lr_scheduler import StepLR

class Dataset:
    def __init__(self, features, outcomes, metadata, mel_spectrograms):
        self.features = features  
        self.outcomes = list(outcomes)
        self.metadata = list(metadata)
        self.mel_spectrograms = list(mel_spectrograms)

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        x = self.features  
        o = self.outcomes[idx]
        md = self.metadata[idx]
        spec = self.mel_spectrograms[idx]
        return x, o, md, spec

    def split(self, nsplits=5):
        from sklearn.model_selection import KFold

        
        kf = KFold(n_splits=nsplits, shuffle=True, random_state=42)

        
        splits = []
        for train_idx, valid_idx in kf.split(np.zeros_like(self.outcomes), self.outcomes):
            train_dataset = Dataset(
                self.features,
                [self.outcomes[i] for i in train_idx],
                [self.metadata[i] for i in train_idx],
                [self.mel_spectrograms[i] for i in train_idx],
            )
            valid_dataset = Dataset(
                self.features,
                [self.outcomes[i] for i in valid_idx],
                [self.metadata[i] for i in valid_idx],
                [self.mel_spectrograms[i] for i in valid_idx],
            )
            splits.append((train_dataset, valid_dataset))
        return splits

def prepare_features(data_folder, model_folder):
    # Find the patient data files.
    patient_files = find_patient_files(data_folder)
    num_patient_files = len(patient_files)

    if num_patient_files == 0:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    print('Extracting features and labels from the Challenge data...')

    outcome_classes = ['Abnormal', 'Normal']
    num_outcome_classes = len(outcome_classes)

    features = list()
    mel_spectrograms = list()
    metadata = list()
    outcomes = list()
    patient_ids = list()
    locations = list()
    globala = 0
    for i in tqdm.tqdm(range(num_patient_files)):

        # Load the current patient data and recordings.
        current_patient_data = load_patient_data(patient_files[i])
        current_recordings = load_recordings(data_folder, current_patient_data)
        
        # Extract features.
        current_features, current_mel_spectrograms = get_features(current_patient_data, current_recordings)
        features.append(current_features)
        
        def apply_windowing(mel_spectrogram, window_size=2, hop_size=2):
            num_frames = mel_spectrogram.shape[-1]
            
            windowed_spectrogram = []
            
            for i in range(0, num_frames - window_size + 1, hop_size):
                window = mel_spectrogram[:, i:i + window_size]
                aggregated_window = window.mean(dim=-1)  # Mean over the time axis (axis=-1)
                windowed_spectrogram.append(aggregated_window)
            
            return torch.stack(windowed_spectrogram, dim=-1)  # Stack the windows along the time axis

        windowed_mel_spectrograms = [apply_windowing(mel) for mel in current_mel_spectrograms]
        
        max_length = max([
            mel.shape[-1]
            for mel in windowed_mel_spectrograms
        ])
        if globala < max_length:
            globala = max_length
        
        # Pad each spectrogram to the same length
        padded_mel_spectrograms = [
            F.pad(mel, (0, 1008 - mel.shape[-1]), mode="constant", value=0)
            for mel in windowed_mel_spectrograms
        ]

        # Ensure there are enough spectrograms in the batch
        num_padding_needed = 6 - len(padded_mel_spectrograms)
        if num_padding_needed > 0:
            padded_mel_spectrograms.extend([torch.zeros_like(padded_mel_spectrograms[0])] * num_padding_needed)

        # Concatenate all spectrograms into a single tensor
        concatenated_mel_spectrograms = torch.cat(padded_mel_spectrograms, dim=0)

        mel_spectrograms.append(concatenated_mel_spectrograms)
 
        current_metadata = current_features
        
        metadata.append(current_metadata)

        current_outcome = np.zeros(num_outcome_classes, dtype=int)
        outcome = get_outcome(current_patient_data)
        if outcome in outcome_classes:
            j = outcome_classes.index(outcome)
            current_outcome[j] = 1
        outcomes.append(current_outcome)
        
    #print(globala)
    mel_spectrograms = [mel.unsqueeze(0) for mel in mel_spectrograms]
    outcomes = np.argmax(outcomes, axis=1)
    return features,metadata,outcomes,mel_spectrograms

def train_outcome_model_features(features, outcomes, metadata, mel_spectrograms, model_folder, ensemble_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    real_cost = 0
    real_AUROC = 0
    epochs = 200
    nepochs = 40
    nsplits = 5

    # Initialize dataset and stratified splits
    dataset = Dataset(features, outcomes, metadata, mel_spectrograms)
    splits = dataset.split(nsplits=nsplits)
    
    
    class WeightedBCELoss(nn.Module):
        def __init__(self, pos_weight):
            super().__init__()
            self.pos_weight = pos_weight

        def forward(self, outputs, targets):
            bce_loss = nn.BCELoss(reduction='none')(outputs, targets)
            weights = (targets * self.pos_weight) + (1 - targets)
            weighted_loss = bce_loss * weights
            return weighted_loss.mean()
    
    for split_idx, (train_dataset, valid_dataset) in enumerate(splits):
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

        mdl = PCGClassifier()
        opt = torch.optim.Adam(mdl.parameters(), lr=0.003)
        scheduler = StepLR(opt, step_size=20, gamma=0.1)  # Reduces LR every 20 epochs by a factor of 0.1
        loss = WeightedBCELoss(pos_weight=5.0)
        
        mdl.to(device)

        mean_cost = 0
        mean_AUROC = 0

        for epoch in range(nepochs):
            mdl.train()
            running_loss = 0.0
            for batch in train_loader:
                x, o, md, spec = batch  # Unpack
                o = o.to(device).float()
                md = md.to(device).float()
                spec = spec.to(device).float()
                
                opt.zero_grad()
      
                z = mdl(spec, md)  # Model forward pass
                
                J = loss(z, o)     # Compute loss
                # Backpropagation
                J.backward()       # Backpropagation
                opt.step()         # Update parameters
                running_loss += J.item()
            
            scheduler.step()
            print(f"Epoch {epoch + 1 + nepochs * split_idx}/{epochs}: Avg Loss: {running_loss / len(train_loader):.4f}")

            # Validation
            mdl.eval()
            outputs, targets = [], []
            with torch.no_grad():
                for batch in valid_loader:
                    x, o, md, spec = batch  # Unpack
                    o = o.to(device).float()
                    md = md.to(device).float()
                    spec = spec.to(device).float()

                    z = mdl(spec, md)
                    targets.append(o.cpu().numpy())
                    outputs.append(z.cpu().numpy())
                    

            # Concatenate validation results
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)

            # Compute AUROC
            AUROC = roc_auc_score(y_true=targets, y_score=outputs)

            # Optimize threshold and compute cost
            score, threshold = optimize_threshold(targets, outputs)
            outcome_classes = ['Abnormal', 'Normal']
            targetsHOT = np.stack([1 - targets, targets], axis=1)
            outputsHOT = outputs > (threshold / 100)
            outputsHOT = np.stack([1 - outputsHOT, outputsHOT], axis=1)
            outcome_cost = compute_cost(targetsHOT, outputsHOT, outcome_classes, outcome_classes)

            mean_cost += outcome_cost
            mean_AUROC += AUROC
            print(f"Validation Epoch {epoch + 1}: AUROC: {AUROC:.4f}, Outcome Cost: {outcome_cost:.4f}")
        
        print(f"Split {split_idx + 1} - Mean Cost: {mean_cost / nepochs:.4f}, Mean AUROC: {mean_AUROC / nepochs:.4f}")
        real_cost += mean_cost / nepochs
        real_AUROC += mean_AUROC / nepochs
    print(f"Final Results: Avg Cost: {real_cost / nsplits:.4f}, Avg AUROC: {real_AUROC / nsplits:.4f}")
    return real_cost / nsplits, real_AUROC / nsplits

min_height = 35.0
max_height = 180.0
min_weight = 2.3
max_weight = 110.8
min_age = 0.5
max_age = 180

def normalize_scalar_minmax(value, min_val, max_val):
    if value is None:
        return 0
    return (value - min_val) / (max_val - min_val)

# Extract features from the data.
def get_features(data, recordings):
    # Extract the age group and replace with the (approximate) number of months for the middle of the age group.
    age_group = get_age(data)

    if compare_strings(age_group, 'Neonate'):
        age = 0.5
    elif compare_strings(age_group, 'Infant'):
        age = 6
    elif compare_strings(age_group, 'Child'):
        age = 6 * 12
    elif compare_strings(age_group, 'Adolescent'):
        age = 15 * 12
    elif compare_strings(age_group, 'Young Adult'):
        age = 20 * 12
    else:
        age = float('nan')

    normalized_age = normalize_scalar_minmax(age, min_age, max_age)
    
    # Extract sex. Use one-hot encoding.
    sex = get_sex(data)

    sex_features = np.zeros(2, dtype=int)
    if compare_strings(sex, 'Female'):
        sex_features[0] = 1
    elif compare_strings(sex, 'Male'):
        sex_features[1] = 1
    
    # Extract height and weight.
    height = get_height(data)
    weight = get_weight(data)
    
    normalized_height = normalize_scalar_minmax(height, min_height, max_height)
    normalized_weight = normalize_scalar_minmax(weight, min_weight, max_weight)
    
    # Extract pregnancy status.
    is_pregnant = get_pregnancy_status(data)

    # Calculate the spectrogram (or Mel-spectrogram) using torchaudio
    # Here, we'll use Mel-spectrogram as an example.
    resample_transform = T.Resample(orig_freq=4000, new_freq=1000)
    mel_transform = T.MelSpectrogram(sample_rate=1000, n_fft=64, hop_length=32, n_mels=32)
    
    mel_spectrograms = []
    for recording in recordings:
        # Convert recording to a PyTorch tensor
        recording_tensor = torch.tensor(recording, dtype=torch.float32)

        # Resample the recording
        resampled_recording = resample_transform(recording_tensor)

        # Compute the Mel-spectrogram
        mel_spectrogram = mel_transform(resampled_recording)

        # Normalize and log-transform the Mel-spectrogram
        mel_log = 10 * torch.log10(mel_spectrogram + 1e-6)
        mel_normalized = (mel_log - mel_log.min()) / (mel_log.max() - mel_log.min())

        mel_spectrograms.append(mel_normalized)

    # Stack features together: Age, Sex, Height, Weight, Pregnancy, etc.
    features = np.hstack((normalized_age, sex_features, normalized_height, normalized_weight, is_pregnant))
    
    # Convert NaNs to zeros
    features = np.nan_to_num(features)
    
    return np.asarray(features, dtype=np.float32), mel_spectrograms