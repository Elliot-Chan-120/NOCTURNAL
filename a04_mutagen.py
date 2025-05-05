import pickle as pkl
import pandas as pd
import random
from pathlib import Path
from padelpy import padeldescriptor
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from b01_utility import *


class MutaGen:
    """
    Generates an optimized molecule based on the results gathered from the previous class' results
    \n Essentially explores nearby chemical space to find alternative and optimal solutions
    \n Local modifications might reveal analogues with better activity, improved pharmacokinetics and reduced toxicity
    """

    def __init__(self, model_name, fingerprint_setting, candidates=10, desired_iterations=100):
        self.mdl_nm = model_name
        self.iterations = desired_iterations
        self.candidates = candidates
        self.fp = fingerprint_setting

        # Verify that all appropriate config keys and folders are present
        # Load cfg
        self.cfg = validate_config()

        # load up files
        self.model_folder = Path(self.cfg['model_folder']) / f"{self.mdl_nm}"
        self.model_file = self.model_folder / f"{self.mdl_nm}.pkl"
        self.model_settings = self.model_folder / f"{self.mdl_nm}_settings.txt"
        self.prediction_file = Path(self.cfg['predictions']) / f"{self.mdl_nm}_predictions.csv"
        self.optimize_database = Path('optimizer_database')

        # files created during run
        self.initial_smiles = self.optimize_database / 'tmp_sml.smi'
        self.mutated_smiles = self.optimize_database / 'mutate_sml.smi'
        self.fingerprint_output = None
        self.optimize_predictions = self.optimize_database / 'tmp_comparison.csv'

        # load up machine learning model
        try:
            with open(self.model_file, "rb") as model:
                self._model = pkl.load(model)  # start up machine learning model
        except FileNotFoundError:
            raise RunModelError("Model File missing")
        except (pkl.UnpicklingError, EOFError) as e:
            raise RunModelError(f"Model loading failed or Model is corrupted: {e}")
        except Exception as e:
            raise RunModelError(f"Unexpected error while loading model: {e}")

        # load up machine learning model settings
        settings = []
        try:
            with open(self.model_settings, 'r') as f:
                for line in f:
                    settings.append(line.strip())
        except FileNotFoundError:
            raise RunModelError(f"Model settings file not found: {self.model_settings}")
        except Exception as e:
            raise RunModelError(f"Error reading settings file: {e}")

        self.settings = settings


    def init_optimize(self):
        """
        initializes the sequence of functions that seeks to optimize a list of SMILES formatted chemicals to obtain greater pIC50 values
        """
        try:
            df  = pd.read_csv(self.prediction_file)
        except Exception as e:
            raise OptimizeError(f"Error loading prediction file: {e}")

        max_row = df.loc[df['pIC50'].idxmax()]
        starting_smiles = max_row['SMILES']
        starting_score = max_row['pIC50']


        starting_mol = Chem.MolFromSmiles(starting_smiles)
        if starting_mol is not None:
            # initialize starting file -> smi file containing rows of the same starting smiles with its pIC50
            with open(self.initial_smiles, 'w') as f:
                for i in range(self.candidates):
                    f.write(f"{starting_smiles}\t{starting_score}\n")

            # optimize the highest scoring SMILES chemical
            self.optimize_control()
        else:
            raise OptimizeError(f"Starting molecule is invalid")


    def optimize_control(self):
        """
        After having everything validated by optimize, it loads up the files and runs random mutation functions on a batch of SMILES
        \n After mutating -> run pIC50 predictions using a previously saved machine learning model
        \n Includes decision-making for mutation choices and progress based on status of current SMILES and score plateaus
        """
        df = pd.read_csv(self.initial_smiles, sep='\t', names=['SMILES', 'pIC50'])
        base_smiles = df['SMILES'].tolist()
        base_score = df['pIC50'].tolist()
        print("Starting SMILES")
        print(base_smiles)

        keep_counter = [0] * self.candidates

        for iteration in range(self.iterations):
            print(f"Iteration {iteration} / {self.iterations}")
            # reset new_smiles list for each iteration
            print(keep_counter)
            new_smiles = []

            # Mutation Round
            for idx in range(self.candidates):
                mutant = self.random_mutation(base_smiles[idx])
                new_smiles.append(mutant)

            # Write Mutations into New smile file
            try:
                with open(self.mutated_smiles, 'w') as f:
                    for i in range(self.candidates):
                        f.write(f"{new_smiles[i]}\n")
            except Exception as e:
                raise OptimizeError(f"Failed to write temporary smiles file: {e}")

            # generate fingerprints for the new molecules
            # fills in missing columns with 0s and reorders them to match training method
            self.fingerprinter()

            # predict pIC50 values for the new mutant molecules
            new_score = self.predict()

            # Compare scores and update base molecules if better
            # check that score has been increased by a certain amount and if it passes oral bioavailability test
            for idx in range(min(len(new_score), len(base_score), len(new_smiles))):
                if keep_counter[idx] >= 3:
                    if (new_score[idx] - base_score[idx]) >= -0.5 and (self.lipinski_check(new_smiles[idx]) >= 0):
                        base_score[idx] = new_score[idx]
                        base_smiles[idx] = new_smiles[idx]
                        keep_counter[idx] = 0
                    else:
                        keep_counter[idx] += 1
                        print('still stuck')
                elif (new_score[idx] - base_score[idx]) >= 0.05 and (self.lipinski_check(new_smiles[idx]) >= 2):
                    base_score[idx] = new_score[idx]
                    base_smiles[idx] = new_smiles[idx]
                else:
                    print('keeping original')
                    keep_counter[idx] += 1
                print(f"{base_smiles[idx]}: {base_score[idx]} -- Retain_Count = {keep_counter[idx]}")

            # rewrite file again with new pIC50 values next to the molecule
            try:
                with open(self.initial_smiles, 'w') as f:
                    for i in range(self.candidates):
                        f.write(f"{base_smiles[i]}\t{base_score[i]}\n")
            except Exception as e:
                raise OptimizeError(f"Failed to write temporary smiles file: {e}")

        final_df = pd.DataFrame({'Optimized SMILES Candidate': base_smiles, 'pIC50 Values': base_score})

        final_df.to_csv(Path(self.cfg['predictions']) / f'{self.model_folder}_optimized_molecules.csv')


    def random_mutation(self, smiles):
        """
        Randomly Mutates a SMILES
        \n Flow: input SMILES -> converted to mol item -> mol item is mutated -> converted back to SMILES and returned
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        mol_edit = Chem.RWMol(mol)
        mutated_mol = None

        atom_indices = list(range(mol.GetNumAtoms()))
        if not atom_indices:
            return Chem.MolToSmiles(mol)

        frags =[
            Chem.MolFromSmiles('C'),
            Chem.MolFromSmiles('CC'),
            Chem.MolFromSmiles('CN'),
            Chem.MolFromSmiles('O'),
            Chem.MolFromSmiles('N'),
            Chem.MolFromSmiles('F'),
            Chem.MolFromSmiles('C(=O)O'),
            Chem.MolFromSmiles('CO'),
            Chem.MolFromSmiles('S'),
        ]

        if len(smiles) >= 3:
            mutation_type = random.choice(["add_group", "replace", "remove"])
        else:
            mutation_type = random.choice(["add_group", "replace"])

        if mutation_type == "add_group":
            insert_group = random.choice(list(frags))

            idx = random.choice(atom_indices)
            atom = mol_edit.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "H" or not self.valence_check(atom):
                return Chem.MolToSmiles(mol)  # don't bind Hydrogen

            combination = Chem.CombineMols(mol_edit, insert_group)
            mole = Chem.EditableMol(combination)
            mol_atoms = mol_edit.GetNumAtoms()

            mole.AddBond(idx, mol_atoms, Chem.BondType.SINGLE)
            mutated_mol = mole.GetMol()


        elif mutation_type == "replace":
            idx_to_change = random.choice(atom_indices)
            atom = mol_edit.GetAtomWithIdx(idx_to_change)
            current_symbol = atom.GetSymbol()
            atom_list = ["C", "N", "O", "F", "Cl", "Br", "S"]
            if current_symbol in atom_list:
                atom_list.remove(current_symbol)

            # attempt 10 replacements
            for x in range(10):
                new_symbol = random.choice(atom_list)
                new_atomicnum = Chem.GetPeriodicTable().GetAtomicNumber(new_symbol)
                atom.SetAtomicNum(new_atomicnum)
                if self.valence_check(atom):
                    break
            else:
                return Chem.MolToSmiles(mol)

            mutated_mol = mol_edit.GetMol()

        elif mutation_type == "remove":
            if len(atom_indices) <= 1:
                return mol
            idx = random.choice(atom_indices)
            mol_edit.RemoveAtom(idx)
            mutated_mol = mol_edit.GetMol()


        try:
            Chem.SanitizeMol(mutated_mol)
            mutated_smiles = Chem.MolToSmiles(mutated_mol)

            if "." in mutated_smiles:
                frags = mutated_smiles.split('.')
                # keep larget fragment
                largest_fragment = max(frags, key=len)
                if len(largest_fragment) < 3:
                    return largest_fragment

        except Exception as e:
            print(f"{smiles} failed to be sanitized, keeping original: {e}")
            return smiles   # return the original if sanitization fails

        return mutated_smiles


    def fingerprinter(self):
        xml_path = Path(self.cfg['padel_xmls'])
        fingerprint_descriptortypes = self.cfg['settings'][self.fp]

        self.fingerprint_output = self.optimize_database / f"new_fingerprint_output_file.csv"

        # Graceful degradation for PaDEL
        try_limit = self.cfg['try_limit']
        for attempt in range(try_limit):
            try:
                padeldescriptor(mol_dir=self.mutated_smiles,
                                d_file=self.fingerprint_output,
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


    def predict(self):
        """
        Uses loaded machine learning model to generate predictions on a list of SMILES
        """

        # load up temporary prediction file
        try:
            input_df = pd.read_csv(self.fingerprint_output, index_col=0)
        except Exception as e:
            raise RunModelError(f"Error reading DataFrame: {e}")


        # quick data integrity check and fill in columns with 0s if the column is empty
        missing_cols = [col for col in self.settings if col not in input_df]
        if missing_cols:
            filler_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
            input_df = pd.concat([input_df, filler_df], axis=1)

        # reorder columns to match training order
        input_df = input_df[self.settings]

        #  filter the input dataframe by the settings columns - used .to_numpy method to avoid a warning since the model was trained on a numpy array
        match_settings = input_df[self.settings].to_numpy()

        try:
            prediction = self._model.predict(match_settings)
        except ValueError as e:
            raise RunModelError(f"Prediction failed: Input may contain shape mismatch, NaN values, or invalid data types"
                                f"\n Details: {e}")
        except Exception as e:
            raise RunModelError(f"Unexpected error during prediction: {e}")

        return prediction


    @staticmethod
    def lipinski_check(smiles):
        """
        Generates dataframe containing relevant information on lipinski rules
        \n mol. weight, octanol water part., H-bond donors and acceptors
        """
        mol = Chem.MolFromSmiles(smiles)

        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
        except Exception as e:
            raise OptimizeError(f"Error calculating Lipinski Descriptors: {e}")

        rules_passed = 0
        if mw <= 500: rules_passed += 1
        if logp <= 5: rules_passed += 1
        if h_donors <=5: rules_passed += 1
        if h_acceptors <= 10: rules_passed += 1

        return rules_passed

    @staticmethod
    def valence_check(atom):
        return atom.GetImplicitValence() > 0 and atom.GetExplicitValence() < Chem.GetPeriodicTable().GetDefaultValence(
            atom.GetAtomicNum())
