import pickle as pkl
import pandas as pd
from pathlib import Path
from padelpy import padeldescriptor


# okay, so basically you need the descriptor list you originally trained the machine learning model with
# load up the model w/ pkl
# generate fingerprints.csv of your input data using padel
# load up the settings you trained your model with -> pubchem columns
# load model.predict(fingerprints.csv[model_settings])
# model will only accept the same parameters and number of inputs you originally trained it on
# therefore, edit your input fingerprints to contain only the columns that the model was originally trained on

class RunModel:

    def __init__(self, model_name, input_smiles_filename, fingerprint):
        self.mdl_nm = model_name
        self.inp_smi = input_smiles_filename
        self.fp = fingerprint

        # NAVIGATION
        # 1. folders
        self.model_folder = Path("ML_Models")
        self.input_folder = Path("_Input_Folder")
        self.input_folder_fpdt = Path(self.input_folder / "input_fingerprint_data")
        self.result_predictions = Path("_Predictions")

        # 2. input files
        self.model = self.model_folder / f"{self.mdl_nm}.pkl"
        self.model_settings = self.model_folder / f"{self.mdl_nm}_settings.txt"
        self.input_smi_file = self.input_folder / f"{self.inp_smi}.smi"

        # 3. output files
        self.valid_smile = self.input_folder / f"VALID_{self.inp_smi}.smi"
        self.inp_fingerprint = None

        # DATA
        self.prediction_list = []

    def validate_smiles(self):
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
        with open(self.model_settings, 'r') as f:
            for line in f:
                settings.append(line.strip())

        return settings


    def fingerprinter(self):
        fp = {
            'AtomPairs2D': 'AtomPairs2DFingerprinter.xml',
            'AtomPairs2DCount': 'AtomPairs2DFingerprintCount.xml',
            'CDK': 'Fingerprinter.xml',
            'CDKextended': 'ExtendedFingerprinter.xml',
            'CDKgraphonly': 'GraphOnlyFingerprinter.xml',
            'EState': 'EStateFingerprinter.xml',
            'KlekotaRoth': 'KlekotaRothFingerprinter.xml',
            'KlekotaRothCount': 'KlekotaRothFingerprintCount.xml',
            'MACCS': 'MACCSFingerprinter.xml',
            'PubChem': 'PubchemFingerprinter.xml',
            'Substructure': 'SubstructureFingerprinter.xml',
            'SubstructureCount': 'SubstructureFingerprintCount.xml'
        }  # dictionary containing descriptor-obtaining methods as key and xml filename as value

        xml_path = "padel_fp_xmls/"
        fingerprint_output_file = ''.join([self.fp, '.csv'])
        fingerprint_descriptortypes = fp[self.fp]

        self.inp_fingerprint = self.input_folder_fpdt / fingerprint_output_file

        padeldescriptor(mol_dir=self.valid_smile,
                        d_file=self.inp_fingerprint,
                        descriptortypes=xml_path + fingerprint_descriptortypes,
                        detectaromaticity=True,
                        standardizenitro=True,
                        standardizetautomers=True,
                        threads=2,
                        removesalt=True,
                        log=True,
                        fingerprints=True)

    def run_predictions(self):
        # load model.predict(fingerprints.csv[model_settings])
        # validates the smiles file, tabbing and inserting identifiers in the event none are there
        self.validate_smiles()

        # calls on fingerprinter: using the validated smiles file it gets a fingerprint file for all compounds submitted
        self.fingerprinter()

        input_df = pd.read_csv(self.inp_fingerprint, index_col=0)

        with open(self.model, "rb") as model:
            load_model = pkl.load(model)  # start up machine learning model

        #  this gets the model settings as a list so that the input fingerprint columns are aligned with the model's
        settings = self.open_settings_txt()

        #  filter the input dataframe by the settings columns - used .to_numpy method to avoid a warning since the model was trained on a numpy array
        match_settings = input_df[settings].to_numpy()
        prediction = load_model.predict(match_settings)

        self.prediction_list = prediction

        self.prediction_file()

    def prediction_file(self):

        print("Generating predictions...")

        molecular_IDs = []
        smiles = []
        # open the smiles file
        with open(self.valid_smile, 'r') as f:
            for line in f:
                sects = line.strip().split()
                smiles.append(sects[0])
                molecular_IDs.append(sects[1])

        result_output_dataframe = pd.DataFrame({
            'Molecule ID': molecular_IDs,
            'SMILES': smiles,
            'pIC50': self.prediction_list
        })

        # Generate files
        result_output_dataframe.to_csv(self.result_predictions / f"{self.mdl_nm}_predictions.csv")
        self.datavis()
        print("Prediction file and pIC50 comparison barchart generated: check dir _Predictions")


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
        