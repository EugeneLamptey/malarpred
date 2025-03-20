import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs



def compute_morganfps(smile_array):
    #this function takes an array containing a single or a series of smile strings and returns an array of ExplicitBitVect type
    if(not isinstance(smile_array, list)):
        raise TypeError("input parameter should be an array")
    mols = get_mols_from_array(smile_array)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in mols]
    return fps


def get_mols_from_array(smile_array):
    #this function takes in an array containing a single smile string or a series of smile string and return an array of mols
    mols = [Chem.MolFromSmiles(smi) for smi in smile_array]
    return mols

def get_tanimomto_similarities(fp1, fp2):
    #this function takes in two bitVects as input
    return DataStructs.cDataStructs.TanimotoSimilarity(fp1, fp2)

def get_bulk_tanimoto_distance(fp1, list_fp):
    similarities = get_bulk_tanimoto_similarities(fp1, list_fp)
    return list(map(lambda p: 1-p, similarities))

def get_bulk_tanimoto_similarities(fp1, list_fp):
    #the first parameter of this function is a bitVect and the second parameter is a list of bitVect
    return DataStructs.cDataStructs.BulkTanimotoSimilarity(fp1, list_fp)