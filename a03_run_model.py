import pickle as pkl

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pathlib import Path
from padelpy import padeldescriptor

from b01_utility import *


class RunModel:
    """
    Loads up ML model of choice and predicts input smiles-formatted compounds' pIC50 for the target protein the ML model was originally trained on
    """
    def __init__(self, model_name, input_smiles_filename):
        self.mdl_nm = model_name
        self.inp_smi = input_smiles_filename


        # Verify that all appropriate config keys and folders are present
        # Load cfg
        self.cfg = validate_config()

        # Get fingerprint settings
        self.fp = get_fingerprint(self.cfg, self.mdl_nm)

        # NAVIGATION
        # 1. folders
        self.model_folder = Path(self.cfg['model_folder']) / f"{self.mdl_nm}"    # accesses target ML folder containing its pkl file and settings
        self.input_folder = Path(self.cfg['input_folder'])
        self.input_folder_fpdt = self.input_folder / self.cfg['input_fp_folder']
        self.result_predictions = Path(self.cfg['predictions'])

        # 2. input files
        self.model = self.model_folder / f"{self.mdl_nm}.pkl"
        self.model_settings = self.model_folder / f"{self.mdl_nm}_settings.txt"
        self.input_smi_file = self.input_folder / f"{self.inp_smi}.smi"

        # 3. output files
        self.valid_smile = self.input_folder / f"VALID_{self.inp_smi}.smi"
        self.inp_fingerprint = None

        # DATA
        self.prediction_list = []
        self.settings_length = None


    def run_predictions(self):
        """Predicts pIC50 values of input SMILES"""

        #validates SMILES file, tabbing and inserting identifiers in the event none are there
        self.validate_smiles()

        # calls on fingerprinter: using the validated smiles file it gets a fingerprint file for all compounds submitted
        self.fingerprinter()

        try:
            input_df = pd.read_csv(self.inp_fingerprint, index_col=0)
        except Exception as e:
            raise RunModelError(f"Error reading DataFrame: {e}")

        try:
            with open(self.model, "rb") as model:
                load_model = pkl.load(model)  # start up machine learning model
        except FileNotFoundError:
            raise RunModelError("Model File missing")
        except (pkl.UnpicklingError, EOFError) as e:
            raise RunModelError(f"Model loading failed or Model is corrupted: {e}")
        except Exception as e:
            raise RunModelError(f"Unexpected error while loading model: {e}")

        #  this gets the model settings as a list so that the input fingerprint columns are aligned with the model's
        settings = self.open_settings_txt()

        # quick data integrity check to see if any column in settings isn't present in the broader input_df
        missing_cols = [col for col in settings if col not in input_df]
        if missing_cols:
            raise DataProcessingError(f"Input file is missing critical columns:{missing_cols}")

        #  filter the input dataframe by the settings columns - used .to_numpy method to avoid a warning since the model was trained on a numpy array
        match_settings = input_df[settings].to_numpy()

        try:
            prediction = load_model.predict(match_settings)
        except ValueError as e:
            raise RunModelError(f"Prediction failed: Input may contain shape mismatch, NaN values, or invalid data types"
                                f"\n Details: {e}")
        except Exception as e:
            raise RunModelError(f"Unexpected error during prediction: {e}")

        self.prediction_list = prediction

        try:
            self.prediction_file()
        except Exception as e:
            raise RunModelError(f"Prediction file save failure: {e}")


    def validate_smiles(self):
        """
        Handles invalid smiles entries: if names are invalid .i.e contain spaces or are missing
        This function will either join improperly formatted IDs together, or insert filler compound IDs
        """

        smiles_list = []
        identity_list = []
        counter = 1

        with open(self.input_smi_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                items = line.split()
                if items:
                    smile = items[0]

                    if len(items) > 1:
                        identity = ''.join(items[1:])
                    else:
                        identity = f"compound_{counter}"
                        counter += 1

                    smiles_list.append(smile)
                    identity_list.append(identity)

        validated_smiles_df = pd.DataFrame({
            'SMILES': smiles_list,
            'ID': identity_list
        })

        # generate validated smiles folder - DONE
        validated_smiles_df.to_csv(self.valid_smile, sep="\t", index=False, header=False)


    def open_settings_txt(self):
        settings = []

        try:
            with open(self.model_settings, 'r') as f:
                for line in f:
                    settings.append(line.strip())
        except FileNotFoundError:
            raise RunModelError(f"Model settings file not found: {self.model_settings}")
        except Exception as e:
            raise RunModelError(f"Error reading settings file: {e}")

        return settings


    def fingerprinter(self):
        xml_path = Path(self.cfg['padel_xmls'])
        fingerprint_output_file = ''.join([self.fp, '.csv'])
        fingerprint_descriptortypes = self.cfg['settings'][self.fp]

        self.inp_fingerprint = self.input_folder_fpdt / fingerprint_output_file

        # Graceful degradation for PaDEL
        try_limit = self.cfg['try_limit']
        for attempt in range(try_limit):
            try:
                padeldescriptor(mol_dir=self.valid_smile,
                                d_file=self.inp_fingerprint,
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


    def prediction_file(self):
        print("Generating predictions...")

        molecular_IDs = []
        smiles = []
        # open the smiles file
        try:
            with open(self.valid_smile, 'r') as f:
                for line in f:
                    sects = line.strip().split()
                    smiles.append(sects[0])
                    molecular_IDs.append(sects[1])
        except Exception as e:
            raise RunModelError(f"Unable to load file {self.valid_smile}: {e}")

        result_output_dataframe = pd.DataFrame({
            'Molecule ID': molecular_IDs,
            'SMILES': smiles,
            'pIC50': self.prediction_list
        })

        try:
            result_output_dataframe.to_csv(self.result_predictions / f"{self.mdl_nm}_predictions.csv")
        except Exception as e:
            raise RunModelError(f"Prediction CSV file save failure: {e}")

        print("Prediction file generated: check dir predictions")

        self.datavis()


    def datavis(self):
        """
        outputs pIC50 comparison barchart
        """
        df = pd.read_csv(self.result_predictions / f"{self.mdl_nm}_predictions.csv")

        sns.set_theme(color_codes=True)
        sns.set_style("white")

        ax = sns.barplot(x=df['Molecule ID'], y=df['pIC50'],  edgecolor='white', linewidth=0.7)
        ax.set_title(f"{self.mdl_nm} Predictions for {self.inp_smi}", fontsize='medium')

        ax.set_xlabel('Molecule Candidates', fontsize='large', fontweight='bold')
        ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')

        ax.figure.set_size_inches(5, 5)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.result_predictions / f"{self.mdl_nm}_candidate_pIC50.pdf")
        plt.close()
