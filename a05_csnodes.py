import pickle
import multiprocessing
import os
from multiprocessing import Pool
from pathlib import Path

from b01_utility import *

from rdkit import Chem
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
from rdkit.Chem import rdFMCS
from rdkit import DataStructs

import pandas as pd

# Adapted from code by Vincent F. Scalfani (BSD 3-Clause License)
# Original Copyright (c) 2022, Vincent F. Scalfani
# Modifications made by Elliot Chan, 2025
# Modification list: modularized the non-function calculation steps into a class / function toolset which are called on together by the function "csn_dataprocessor"
# continued: added error handling

def tan_similarity(nodes):
    """generate dict containing all node pair subsets and their smiles + mols + tanimoto similarity values"""
    from itertools import combinations

    smis = []
    for key, value in nodes.items():
        try:
            if Chem.MolFromSmiles(key) is not None:
                smis.append(key)
            else:
                print(f"Invalid SMILES structure skipped: {key}")
        except Exception as e:
            raise CSNDataError(f"Error processing SMILES {key}: {e}")

    # generate all combinations of smiles pairs
    smis_subsets = list(combinations(smis, 2))
    # smis_subsets contains pair tuples of all smiles pair combinations

    # generate dictionary containing smiles pairs and edge data
    subsets = {}
    for i, (smi1, smi2) in enumerate(smis_subsets):
        field = {
            "smi1": smi1,
            "smi2": smi2,
            "mol1": Chem.MolFromSmiles(smi1),
            "mol2": Chem.MolFromSmiles(smi2)
        }
        subsets[i] = field

    # Compute Tanimoto Similarity using RDKit fp calculations
    for key, value in subsets.items():
        try:
            if value['mol1'] is None or value['mol2'] is None:
                subsets[key].update({"tan_similarity": 0.0})
                continue

            # fingerprints from rdkit
            fp1 = Chem.RDKFingerprint(value['mol1'])
            fp2 = Chem.RDKFingerprint(value['mol2'])

            if fp1 is None or fp2 is None:
                print(f"Failed to generate fingerprint pair: {value['smi1']} | {value['smi2']}")
                continue

            tan_sim = round(DataStructs.TanimotoSimilarity(fp1, fp2), 3)
            subsets[key].update({"tan_similarity": tan_sim})

        except Exception as e:
            print(f"Error calculating Tanimoto Similarity for {key}: {e}")

    return subsets


def tc_mcs(mol1, mol2, key):
    if mol1 is None or mol2 is None:
        return key, 0.0
    mcs = rdFMCS.FindMCS([mol1, mol2], timeout=10)

    # get number of common bonds
    mcs_bonds = mcs.numBonds

    # get number of bonds for each, default is only heavy atom bonds
    mol1_bonds = mol1.GetNumBonds()
    mol2_bonds = mol2.GetNumBonds()

    # compute MCS-based Tanimoto
    tan_mcs = mcs_bonds / (mol1_bonds + mol2_bonds - mcs_bonds)

    return key, tan_mcs


class CSNodes:
    """Creates node data using smiles data gathered from ML model results"""
    def __init__(self, model_name, network_type):
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")

        # get validated config
        self.cfg = validate_config()

        # second validation of model_name -> make sure it has already run on MutaGen
        self.model_name = model_name
        self.model_folder = Path(self.cfg['model_folder'])
        folders = os.listdir(self.model_folder)
        if self.model_name not in folders:
            raise CSNDataError(f"{self.model_name} does not exist in the model storage: Check that this is a valid stored model that MutaGen (a04) has used and results saved")

        self.network_type = network_type

        # make database folder within database
        # make folder to hold the numerous assessment files
        self.storage = Path(self.cfg['database']) / self.cfg['network_folder'] / f"{self.model_name}_network_database"  # graph folder
        self.storage.mkdir(parents=True, exist_ok=True)


        # Detect map type
        if self.network_type == "optima":
            self.filepath = Path(self.cfg['predictions']) / f"{self.model_name}_local_optima_compounds.csv"
            self.smiles_str = 'Optima SMILES'
            self.potency_str = 'pIC50 Values'
        elif self.network_type == "optimized":
            self.filepath = Path(self.cfg['predictions']) / f"{self.model_name}_optimized_compounds.csv"
            self.smiles_str = 'Target SMILES'
            self.potency_str = 'pIC50 Values'
        else:
            raise ValueError("Invalid Chemical Space Network Type Detected. Select between: Optimized / Optima")


# need to generate node data -> keys = SMILES, values = pIC50 values
# filter out structures that are fragmented
# tanimoto similarity calculated by fingerprints from rdkit
    def get_nodes(self):
        """
        Make starting node dictionary
        :return: dict -> smiles:pIC50
        """
        # Read CSV + handle errors -> need to make sure im not handling an empty dataframe
        try:
            if not self.filepath.exists():
                raise CSNDataError(f"Data file not found: {self.filepath}")
        except Exception as e:
            print(f"Unexpected error at a05_csndata module's get_nodes class function: {e}")


        try:
            file_df = pd.read_csv(self.filepath)
        except pd.errors.EmptyDataError:
            raise CSNDataError(f"CSV file empty: {self.filepath}")
        except pd.errors.ParserError as e:
            raise CSNDataError(f"Error parsing CSV file: {e}")
        except PermissionError:
            raise CSNDataError(f"Permission denied accessing: {self.filepath}")
        except Exception as e:
            raise CSNDataError(f"Error reading CSV file: {e}")

        # check for empty dataframe
        if file_df.empty:
            raise CSNDataError(f"No data found in csv file: try running {self.model_name} until you obtain {self.network_type} molecules")

        df_smiles = file_df[self.smiles_str]
        df_potency = file_df[self.potency_str]

        # make node dict
        nodes = {}
        for idx in range(len(df_smiles)):
            if '.' not in df_smiles[idx]:
                nodes.update({df_smiles[idx]: df_potency[idx]})

        return nodes

    def targeted_save(self, data, datatype):

        savepath = Path(self.storage) / f"{self.model_name}_{self.network_type}_{datatype}.pkl"

        try:
            if savepath.exists():
                backup_path = savepath.with_suffix('.pkl.backup')
                savepath.rename(backup_path)

            with open(savepath, 'wb') as output:
                pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

        except PermissionError:
            raise CSNDataError(f"{savepath} access denied")
        except OSError as e:
            raise CSNDataError(f"OS error saving file: {e}")
        except Exception as e:
            raise CSNDataError(f"Error saving {datatype}: {e}")


def csn_dataprocessor(model_name, network_type):
    """
    Need to run this twice, with both optimized and optima as the network_type settings in order to get the full dataset
    :return: 2 pkl files containing subset and node data dictionaries -> subsets contain tanimoto similarity values
    """
    item = CSNodes(model_name, network_type)
    node_data = item.get_nodes()

    subsets = tan_similarity(node_data)
    if not subsets:
        raise CSNDataError("No molecular pairs generated for similarity calculations")

    num_cpus = max(1, multiprocessing.cpu_count() - 2)

    mol_tuples = []
    for key, value in subsets.items():
        mol_tuples.append((value['mol1'], value['mol2'], key))

    if __name__ == '__main__':
        with Pool(num_cpus) as p:
            star_map = p.starmap(tc_mcs, mol_tuples)
        for key, tan_mcs in star_map:
            subsets[key].update({"tan_mcs": round(tan_mcs, 3)})

        item.targeted_save(subsets, "subsets")
        item.targeted_save(node_data, "node_data")
