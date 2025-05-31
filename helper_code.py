import os, numpy as np, scipy as sp, scipy.io, scipy.io.wavfile

# Compute a binary confusion matrix, where the columns are the expert labels and the rows are the classifier labels.
def compute_confusion_matrix(labels, outputs):
    assert(np.shape(labels)[0] == np.shape(outputs)[0])
    assert(all(value in (0, 1, True, False) for value in np.unique(labels)))
    assert(all(value in (0, 1, True, False) for value in np.unique(outputs)))

    num_patients = np.shape(labels)[0]
    num_label_classes = np.shape(labels)[1]
    num_output_classes = np.shape(outputs)[1]

    A = np.zeros((num_output_classes, num_label_classes))
    for k in range(num_patients):
        for i in range(num_output_classes):
            for j in range(num_label_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A

# Define total cost for algorithmic prescreening of m patients.
def cost_algorithm(m):
    return 10*m

# Define total cost for expert screening of m patients out of a total of n total patients.
def cost_expert(m, n):
    return (25 + 397*(m/n) -1718*(m/n)**2 + 11296*(m/n)**4) * n

# Define total cost for treatment of m patients.
def cost_treatment(m):
    return 10000*m

# Define total cost for missed/late treatement of m patients.
def cost_error(m):
    return 50000*m

# Compute Challenge cost metric.
def compute_cost(labels, outputs, label_classes, output_classes):
    # Define positive and negative classes for referral and treatment.
    positive_classes = ['Abnormal']
    negative_classes = ['Normal']

    # Compute confusion matrix.
    A = compute_confusion_matrix(labels, outputs)

    # Identify positive and negative classes for referral.
    idx_label_positive = [i for i, x in enumerate(label_classes) if x in positive_classes]
    idx_label_negative = [i for i, x in enumerate(label_classes) if x in negative_classes]
    idx_output_positive = [i for i, x in enumerate(output_classes) if x in positive_classes]
    idx_output_negative = [i for i, x in enumerate(output_classes) if x in negative_classes]

    # Identify true positives, false positives, false negatives, and true negatives.
    tp = np.sum(A[np.ix_(idx_output_positive, idx_label_positive)])
    fp = np.sum(A[np.ix_(idx_output_positive, idx_label_negative)])
    fn = np.sum(A[np.ix_(idx_output_negative, idx_label_positive)])
    tn = np.sum(A[np.ix_(idx_output_negative, idx_label_negative)])
    total_patients = tp + fp + fn + tn

    # Compute total cost for all patients.
    total_cost = cost_algorithm(total_patients) \
        + cost_expert(tp + fp, total_patients) \
        + cost_treatment(tp) \
        + cost_error(fn)

    # Compute mean cost per patient.
    if total_patients > 0:
        mean_cost = total_cost / total_patients
    else:
        mean_cost = float('nan')

    return mean_cost

def optimize_threshold(targets,outputs,plot=True):
    result = []
    for threshold in range(0,100):
        outcome_classes = ['Abnormal', 'Normal']
        targetsHOT = np.stack([1 - targets, targets], axis=1)

        outputsHOT = outputs > threshold/100
        outputsHOT = np.stack([1 - outputsHOT, outputsHOT], axis=1)

        outcome_cost = compute_cost(targetsHOT, outputsHOT, outcome_classes, outcome_classes)
        
        result.append(outcome_cost)
    return np.min(result),np.argmin(result)

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Compare normalized strings.
def compare_strings(x, y):
    try:
        return str(x).strip().casefold()==str(y).strip().casefold()
    except AttributeError: # For Python 2.x compatibility
        return str(x).strip().lower()==str(y).strip().lower()

# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.txt':
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames

# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency

# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings

# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id

# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations

# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[2])
            except:
                pass
        else:
            break
    return frequency

# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i==0:
            pass
        elif 1<=i<=num_locations:
            locations.append(entries[0])
        else:
            break
    return locations

# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age

# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex

# Get height from patient data.
def get_height(data):
    height = None
    for l in data.split('\n'):
        if l.startswith('#Height:'):
            try:
                height = float(l.split(': ')[1].strip())
            except:
                pass
    return height

# Get weight from patient data.
def get_weight(data):
    weight = None
    for l in data.split('\n'):
        if l.startswith('#Weight:'):
            try:
                weight = float(l.split(': ')[1].strip())
            except:
                pass
    return weight

# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                is_pregnant = bool(sanitize_binary_value(l.split(': ')[1].strip()))
            except:
                pass
    return is_pregnant

# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for l in data.split('\n'):
        if l.startswith('#Outcome:'):
            try:
                outcome = l.split(': ')[1]
            except:
                pass
    if outcome is None:
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return outcome

# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = str(x).replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return 0

# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = str(x).replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and np.isinf(float(x))):
        return float(x)
    else:
        return 0.0