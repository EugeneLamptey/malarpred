from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pickle
import os
import pandas as pd

#calculate morgan fingerprints for a single input
def compute_morganfps(mol):
    fps = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # Calculate Morgan fingerprints
    fp_array = np.array(fps)   # Convert fingerprint to numpy array
    column_names = ['morgan_' + str(i) for i in range(len(fp_array))]


    data = pd.DataFrame([fp_array], columns=column_names)
    return data

#calculate morgan fingerprint for csv files
def morgan_csv(df_csv):
    data = pd.read_csv(df_csv, header = None)

    def morgan_fps(data):
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in data]  # Calculate Morgan fingerprints for each molecule
        fp_array = [np.array(fp) for fp in fps]   # Convert fingerprints to numpy array
        column_names = ['morgan_' + str(i) for i in range(len(fp_array[0]))]

        data = pd.DataFrame(fp_array, columns = column_names)
        return data

    smiles_list = data[0].to_list()
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    return morgan_fps(mols), data[0]

# create model
def create_model(model_type: str):
    base_file_path = os.path.join('models')
    
    # Define model file names based on the model type
    model_files = {
        'RFC': 'rfc_pipeline.pkl',
        'CTM': 'catboost_pipeline.pkl',
        'GBM': 'gbm_pipeline.pkl'
    }
    
    if model_type not in model_files:
        return "Pick a valid model type"
    
    model_file = os.path.join(base_file_path, model_files[model_type])
    
    # Open and load the model
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    
    return model
    
#compute rfc for a csv file
def csv_result(descriptors, data, model, isWithin):

    activity = model.predict(descriptors)
    pred_proba = model.predict_proba(descriptors)

    smiles = data

    #get confidence scores from models
    confidence = pred_proba[np.arange(len(activity)), activity]

    #column = ['Canonical Smiles', 'Activity', 'Confidence', 'Within AD']
    dataframe = {'Canonical Smiles': smiles, 'Activity': activity, "Confidence": confidence, 'Within AD': isWithin}
    
    df = pd.DataFrame(dataframe)

    # Replace values in the 'Activity' column
    df['Activity'] = df['Activity'].replace({1: 'active', 0: 'inactive'})
    return df

# Define a function to apply CSS styles
def format_smiles(smiles):
    return f'<div class="truncate">{smiles}</div>'