# ML-Capstone-2

# Automatic ECG Diagnosis - Setup Guide

This README provides instructions for setting up the environment for the `automatic-ecg-diagnosis` repository and incorporating custom .pickle files.

## Repository Overview

The `automatic-ecg-diagnosis` repository contains scripts and modules for training and testing neural networks for ECG automatic classification. It is the companion code to the paper "Automatic diagnosis of the 12-lead ECG using a deep neural network" published in Nature Communications.

## Installation

### Prerequisites

The code has been tested with:
- Python 3
- TensorFlow 2.2

### Step 1: Clone the Repository

```bash
git clone https://github.com/antonior92/automatic-ecg-diagnosis.git
cd automatic-ecg-diagnosis
```

### Step 2: Set Up the Environment

#### Option 1: Using pip

1. Create a virtual environment (recommended):
```bash
python -m venv ecg_env
source ecg_env/bin/activate  # On Windows: ecg_env\Scripts\activate
```

2. Install the required packages:
```bash
pip install tensorflow==2.2.0
pip install numpy pandas h5py matplotlib seaborn scikit-learn
```

#### Option 2: Using conda

1. Create a conda environment:
```bash
conda create --name ecg_env python=3.8
conda activate ecg_env
```

2. Install the required packages:
```bash
conda install tensorflow=2.2.0
conda install numpy pandas h5py matplotlib seaborn scikit-learn
```

## Data Structure

The repository expects ECG data in a specific format:

- Input ECG data shape: (N, 4096, 12)
  - N: Number of samples
  - 4096: Signal points sampled at 400Hz (approximately 10 seconds)
  - 12: Different leads in the order {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}

- Signals should be represented as 32-bit floating-point numbers at the scale 1e-4V

## Incorporating Your .pickle Files

To incorporate your .pickle files into the workflow, follow these steps:

### Step 1: Create a Conversion Script

Create a Python script named `convert_pickle.py` in the repository root:

```python
import pickle
import numpy as np
import h5py
import os

# Load your pickle files
def load_pickle_files(pickle_file1, pickle_file2):
    with open(pickle_file1, 'rb') as f1:
        data1 = pickle.load(f1)
    with open(pickle_file2, 'rb') as f2:
        data2 = pickle.load(f2)
    return data1, data2

# Convert and save data to the format expected by the repository
def convert_to_hdf5(data, output_path, dataset_name='tracings'):
    """
    Convert data to HDF5 format expected by the repository.
    
    Ensure data shape is (N, 4096, 12) where:
    - N is the number of samples
    - 4096 is the signal points (padded with zeros if needed)
    - 12 is the number of ECG leads
    """
    # Process data as needed to match the expected format
    # This is placeholder logic - modify according to your data structure
    processed_data = preprocess_data(data)
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset(dataset_name, data=processed_data)
    
    print(f"Data saved to {output_path} with shape {processed_data.shape}")

def preprocess_data(data):
    """
    Preprocess your data to match the expected format:
    - Shape: (N, 4096, 12)
    - Scale: 1e-4V (multiply by 1000 if data is in V)
    - Zero-pad if needed to reach 4096 points
    """
    # This is placeholder logic - modify according to your data structure
    # Example code for zero padding (if your data has fewer than 4096 points)
    N, current_points, leads = data.shape  # Assuming your data is already in this format
    
    if current_points < 4096:
        # Calculate padding size
        pad_size = (4096 - current_points) // 2
        pad_end = 4096 - current_points - pad_size
        
        # Create padded array
        padded_data = np.zeros((N, 4096, leads))
        padded_data[:, pad_size:pad_size+current_points, :] = data
        
        # Scale if needed (example: if data is in V, scale to 1e-4V)
        # padded_data = padded_data * 1000  # Uncomment if needed
        
        return padded_data
    elif current_points > 4096:
        # If data has more points, truncate
        return data[:, :4096, :]
    else:
        # If data already has 4096 points
        return data

# Main execution
if __name__ == "__main__":
    # Specify your pickle file paths
    pickle_file1 = "path/to/your/first_file.pickle"
    pickle_file2 = "path/to/your/second_file.pickle"
    
    # Load data
    data1, data2 = load_pickle_files(pickle_file1, pickle_file2)
    
    # Process and combine data as needed
    # This is placeholder logic - modify according to your data structure
    combined_data = np.concatenate([data1, data2], axis=0)
    
    # Convert and save
    output_path = "data/your_ecg_data.hdf5"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    convert_to_hdf5(combined_data, output_path)
    
    print("Conversion complete!")
```

### Step 2: Adapt the Script to Your Data

The script above contains placeholder logic. You need to adapt it to the specific structure of your .pickle files:

1. Ensure the `preprocess_data` function correctly formats your data to match the repository's expected format:
   - Shape (N, 4096, 12)
   - Correct scaling (multiply by 1000 if your data is in V)
   - Proper lead ordering {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}

2. If your data has annotations/labels, create a CSV file with the appropriate format. The repository expects a CSV with columns for each abnormality type (1dAVb, RBBB, LBBB, SB, AF, ST) with binary values (0 or 1).

### Step 3: Run the Conversion Script

```bash
python convert_pickle.py
```

### Step 4: Use the Data for Training/Inference

After conversion, your data is ready to be used with the repository's scripts:

- For training:
```bash
python train.py data/your_ecg_data.hdf5 data/your_annotations.csv
```

- For generating figures and tables (if your data is for testing):
```bash
python generate_figures_and_tables.py
```

## Model Architecture

The repository implements a deep neural network architecture for ECG classification:

- Input: ECG signal with 12 leads
- Architecture: Residual neural network (ResNet)
- Output: Probabilities for 6 different ECG abnormalities (1dAVb, RBBB, LBBB, SB, AF, ST)

## Troubleshooting

If you encounter issues:

1. Ensure your data is correctly formatted (shape, scale, lead order)
2. Check TensorFlow version compatibility (tested with version 2.2)
3. For older TensorFlow versions, try the tensorflow-v1 branch of the repository

## References

Ribeiro, A.H., Ribeiro, M.H., Paixão, G.M.M. et al. Automatic diagnosis of the 12-lead ECG using a deep neural network. Nat Commun 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4

## Additional Resources

For preprocessing ECG data before using this repository, you may refer to the complementary repository:
https://github.com/antonior92/ecg-preprocessing
