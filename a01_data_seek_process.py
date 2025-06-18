import requests

from chembl_webresource_client.new_client import new_client

import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from padelpy import padeldescriptor

from pathlib import Path

from b01_utility import *


class DataSeekProcess:
    """
    This class will search the chembl website for target protein data, and then investigate a selected target
    \n After data cleaning, removing redundant info, and obtaining a fingerprint csv, it will generate a dataframe csv containing each compound's fingerprint and their pIC50 values to train the machine learning model
    """
    def __init__(self, target_protein, selected_target_index, fingerprint_setting):
        self.target_subject = target_protein
        self.target_idx = selected_target_index
        self.fingerprint = fingerprint_setting

        # Verify that all appropriate config keys and folders are present
        # Load cfg
        self.cfg = validate_config()

        # NAVIGATION
        # database folder
        self.bioact_folder = Path(self.cfg['database'])

        # static files
        self.rawdf_path = self.bioact_folder / self.cfg['rawdf_1']
        self.labeldf_path = self.bioact_folder / self.cfg['labeldf_2']
        self.pIC50df_path = self.bioact_folder / self.cfg['pIC50df_3']
        self.smile_path = self.bioact_folder / self.cfg['smile_data']    # holds smiles
        self.padel_opath = self.bioact_folder / self.cfg['fp_output']    # holds fingerprints
        self.fppIC50df_path = self.bioact_folder / self.cfg['fingerprintdf_4']


    def run(self):
        print("Starting...")

        self.preprocess_1()
        self.process_2()
        self.smile_fingerprinter_3()

        print("Data Processing Complete")


    def preprocess_1(self):
        """Search CHEMBL and obtain data - preprocess, cleaning and saving csv file containing chembl ids, canonical smiles and standard values"""
        try:
            target = new_client.target
            target_query = target.search(self.target_subject)
            targets = pd.DataFrame.from_dict(target_query)

            if not target_query:
                raise ChEMBLAPIError(f"No targets found for query: {self.target_subject}")
        except requests.exceptions.RequestException as e:
            raise ChEMBLAPIError(f"Network Error when attempting to connect to ChEMBL API: {e}")
        except Exception as e2:
            raise ChEMBLAPIError(f"Unexpected Error during query for target: {e2}")

        # handle invalid indexing for target selection
        if self.target_idx >= len(targets):
            raise DataProcessingError(f"Target Index out of range for {len(targets)} targets.")

        selected_target = targets.target_chembl_id[self.target_idx]
        print("Processing Data on Selected Target:", selected_target)

        # obtain bioactivity data reported as IC50 values in nM
        activity = new_client.activity
        result = activity.filter(target_chembl_id=selected_target).filter(standard_type = "IC50")
        raw_df = pd.DataFrame.from_dict(result)
        # file handling check - raw_df
        try:
            raw_df.to_csv(self.rawdf_path, index=False)
        except Exception as e:
            raise DataSeekProcessError(f"Failed to save raw DataFrame: {e}")

        # eliminate compounds with no standard_value -- standardize our df
        standard_df = raw_df[raw_df.standard_value.notna()]

        # Label compounds either as active, inactive or intermediate
        bioactivity_class = []
        for i in standard_df.standard_value:
            if float(i) >= 10000:
                bioactivity_class.append("inactive")
            elif float(i) <= 1000:
                bioactivity_class.append("active")
            else:  # in between 1000 and 10000
                bioactivity_class.append("intermediate")

        # complete pre-processing - we want chembl ids, canonical SMILES and standard values
        selected_columns = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']

        # validate all columns are there
        missing_col = [col for col in selected_columns if col not in raw_df.columns]
        if missing_col:
            raise DataProcessingError(f"Missing required columns in ChEMBL Data: {missing_col}")

        # continue - filter standardized df by taking just the selected_columns data
        dummy_df = standard_df[selected_columns]
        preprocess_df = dummy_df.copy()
        preprocess_df['bioactivity_class'] = bioactivity_class
        # file handling checks
        try:
            preprocess_df.to_csv(self.labeldf_path)
        except Exception as e:
            raise DataSeekProcessError(f"Failed to save labelled DataFrame: {e}")


    def process_2(self):
        """produce dataframe containing lipinski information, normalized and pIC50 values"""
        try:
            preprocess_df = pd.read_csv(self.labeldf_path)
        except FileNotFoundError:
            raise DataSeekProcessError(f"Labelled DataFrame File Missing")
        except Exception as e:
            raise DataSeekProcessError(f"Unexpected Error Loading {self.labeldf_path}: {e}")

        # generate lipinski dataframe - future exploratory data analysis
        lipinski_df = self.lipinski_info(preprocess_df.canonical_smiles)

        dummy_df = pd.concat([preprocess_df, lipinski_df], axis=1)

        # normalize values in dataframe
        norm_df = self.norm_value(dummy_df)

        # calculate pIC50 values from IC50 -> replace those values in dataframe
        # this dataframe now has lipinski, normalized, and pIC50 data
        lip_pic_norm_df = self.pIC50(norm_df)

        process2_df = lip_pic_norm_df[lip_pic_norm_df.bioactivity_class != 'intermediate']

        try:
            process2_df.to_csv(self.pIC50df_path)
        except Exception as e:
            raise DataSeekProcessError(f"Failed to save pIC50-processed DataFrame: {e}")


    def smile_fingerprinter_3(self):
        """Generate molecule smiles and fingerprint file of all filtered compounds for the model to process"""
        try:
            prefingerprint_df = pd.read_csv(self.pIC50df_path)
        except FileNotFoundError:
            raise DataSeekProcessError(f"pIC50 DataFrame File Missing")
        except Exception as e:
            raise DataSeekProcessError(f"Unexpected Error Loading {self.pIC50df_path}: {e}")

        selection = ['canonical_smiles', 'molecule_chembl_id']
        selected_df = prefingerprint_df[selection]

        # create smiles file for padel to process into a fingerprint csv
        try:
            selected_df.to_csv(self.smile_path, sep='\t', index=False, header=False)
        except Exception as e:
            raise DataSeekProcessError(f"Failed to save SMILES file: {e}")

        xml_path = Path(self.cfg['padel_xmls'])
        fingerprint_descriptortypes = self.cfg['settings'][self.fingerprint]

        # in event of errors -> Graceful degradation handling for PaDEL
        try_limit = self.cfg['try_limit']
        for attempt in range(try_limit):
            try:
                padeldescriptor(mol_dir=self.smile_path,
                                d_file= self.padel_opath,
                                descriptortypes=xml_path / fingerprint_descriptortypes,
                                detectaromaticity=True,
                                standardizenitro=True,
                                standardizetautomers=True,
                                threads=2,
                                removesalt=True,
                                log=True,
                                fingerprints=True)
                break  # upon success -> proceed and break out of the loop
            except Exception as e:
                if attempt < try_limit - 1:
                    continue
                raise PaDELProcessError(f"PaDEL molecular fingerprint calculations failed after {try_limit} tries: {e}")

        # X and Y dataframes to be combined
        try:
            fp_X = pd.read_csv(self.padel_opath).drop(columns=['Name'])
            fp_Y = prefingerprint_df['pIC50']
        except FileNotFoundError:
            raise DataSeekProcessError(f"PaDEL output file not found at {self.padel_opath}")
        except Exception as e:
            raise DataSeekProcessError(f"Failed to read PaDEL output file: {e}")

        # data integrity check
        if len(fp_X) != len(prefingerprint_df):
            raise DataProcessingError(f"PaDEL returned {len(fp_X)} compounds, but expected {len(prefingerprint_df)}")

        fingerprint_df = pd.concat([fp_X, fp_Y], axis=1)

        try:
            fingerprint_df.to_csv(self.fppIC50df_path)
        except Exception as e:
            raise DataSeekProcessError(f"Failed to save Molecular Fingerprint + pIC50 DataFrame: {e}")


    @staticmethod
    def lipinski_info(smiles, verbose=False):
        """
        Generates dataframe containing relevant information on lipinski rules
        \n mol. weight, octanol water part., H-bond donors and acceptors
        """
        moldata = []
        for elem in smiles:
            mol = Chem.MolFromSmiles(elem)
            moldata.append(mol)

        baseData = np.arange(1, 1)
        i = 0
        for mol in moldata:   # ignore the errors, the identifiers are there idk what's up
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

            row = np.array([desc_MolWt,
                            desc_MolLogP,
                            desc_NumHDonors,
                            desc_NumHAcceptors])

            if i == 0:
                baseData = row
            else:
                baseData = np.vstack([baseData, row])
            i += 1

        columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
        descriptors = pd.DataFrame(data=baseData, columns=columnNames)

        return descriptors

    @staticmethod
    def pIC50(dataframe):
        """
        works with the standard_value_norm column and converts from IC50 to pIC50
        """
        pic50 = []

        for i in dataframe['standard_value_norm']:  # access and iterate through the standard value norm column

            molar = i * (10 ** -9)  # convert nM to M units
            pic50.append(-np.log10(molar))

        dataframe['pIC50'] = pic50
        x = dataframe.drop('standard_value_norm', axis=1)

        return x

    @staticmethod
    def norm_value(dataframe):
        """
        works with the standard_value_norm column and normalizes all values
        """
        norm = []

        for i in dataframe['standard_value']:
            if i > 100000000:
                i = 100000000
            norm.append(i)

        dataframe['standard_value_norm'] = norm
        norm_df = dataframe.drop('standard_value', axis=1)

        return norm_df
