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
    \n Performs both chemical space exploration and drug optimization, aiming to generate improved drug candidates optimized against target protein with which the ML model was trained on
    \n Local modifications might reveal analogues with better activity, improved pharmacokinetics and reduced toxicity
    """

    def __init__(self, model_name):
        self.mdl_nm = model_name

        # Verify that all appropriate config keys and folders are present
        # Load cfg
        self.cfg = validate_config()
        self.iterations = self.cfg['iterations']
        self.candidates = self.cfg['candidates']

        # Get fingerprint settings
        self.fp = get_fingerprint(self.cfg, self.mdl_nm)

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
        self.start_score = 0


    def init_optimize(self):
        """
        initializes the sequence of functions that seeks to optimize a list of SMILES formatted chemicals to obtain greater pIC50 values
        """
        try:
            df  = pd.read_csv(self.prediction_file)
        except Exception as e:
            raise MutaGenError(f"Error loading prediction file: {e}")

        max_row = df.loc[df['pIC50'].idxmax()]
        starting_smiles = max_row['SMILES']
        starting_score = max_row['pIC50']

        # this is the baseline score for our optimize control function
        # -> seeing which molecules have an increase in pIC50 past a config-defined threshold
        self.start_score = starting_score


        starting_mol = Chem.MolFromSmiles(starting_smiles)
        if starting_mol is not None:
            # initialize starting file -> smi file containing rows of the same starting smiles with its pIC50
            with open(self.initial_smiles, 'w') as f:
                for i in range(self.candidates):
                    f.write(f"{starting_smiles}\t{starting_score}\n")

            # optimize the highest scoring SMILES chemical
            self.optimize_control()
        else:
            raise MutaGenError(f"Starting molecule is invalid")


    def optimize_control(self):
        """
        After having everything validated by optimize, it loads up the files and runs random mutation functions on a batch of SMILES
        \n After mutating -> run pIC50 predictions using the chosen machine learning model
        \n Includes decision-making for mutation choices and progress based on status of current SMILES and score plateaus
        """
        # update this? lots of nesting which isn't easy to follow

        df = pd.read_csv(self.initial_smiles, sep='\t', names=['SMILES', 'pIC50'])
        base_smiles = df['SMILES'].tolist()
        base_score = df['pIC50'].tolist()
        print("Starting SMILES")
        print(base_smiles)
        optima_smiles = []
        optima_scores = []

        target_smiles = []
        target_scores = []

        keep_counter = [0] * self.candidates

        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1} / {self.iterations}")
            # reset new_smiles list for each iteration
            print(keep_counter)
            new_smiles = []

            # Mutation Round
            for idx in range(self.candidates):
                mutant = self.random_mutation(base_smiles[idx])
                new_smiles.append(mutant)

            # Write Mutations into new smile file
            try:
                with open(self.mutated_smiles, 'w') as f:
                    for i in range(self.candidates):
                        f.write(f"{new_smiles[i]}\n")
            except Exception as e:
                raise MutaGenError(f"Failed to write temporary smiles file: {e}")

            # generate fingerprints for the new molecules
            # fills in missing columns with 0s and reorders them to match training method
            self.fingerprinter()

            # predict pIC50 values for the new mutant molecules
            new_score = self.predict()

            # Compare scores and update base molecules if better

            # check that score has been increased by a certain amount and passes the oral bioavailability test
            for idx in range(min(len(new_score), len(base_score), len(new_smiles))):
                # this means that the compound has failed to improve 'x' amount of times and is now considered a local optima
                if keep_counter[idx] >= self.cfg['retain_threshold']:
                    optima_smiles.append(new_smiles[idx]), optima_scores.append(new_score[idx])
                    # ADAPTIVE OPTIMA ESCAPE MECHANISMS
                    # continuously decrease the error threshold by an additional fraction of its original value depending on keep_counter
                    # stricten up the requirements to filter out molecules that retain some oral bioavailability and will produce greater improvements
                    if (new_score[idx] - base_score[idx]) >= (self.cfg['error_threshold'] + (self.cfg['error_threshold'] * (1/keep_counter[idx]))) and (self.lipinski_check(new_smiles[idx]) > 0):
                        base_score[idx] = new_score[idx]
                        base_smiles[idx] = new_smiles[idx]
                        keep_counter[idx] = 0
                    else:
                        keep_counter[idx] += 1
                        print('still stuck')
                # this means the new compound is optimized to or past the point of our goal -> ADD IT TO THE LIST!
                elif new_score[idx] - self.start_score >= self.cfg['target_increase']:
                    target_smiles.append(new_smiles[idx]), target_scores.append(new_score[idx])
                    base_score[idx] = new_score[idx]
                    base_smiles[idx] = new_smiles[idx]
                # if the new compound meets our success threshold - added moving it to the next iteration for increased exploration around those molecules
                elif (new_score[idx] - base_score[idx]) >= self.cfg['success_threshold'] and (self.lipinski_check(new_smiles[idx]) >= 2):
                    base_score[idx] = new_score[idx]
                    base_smiles[idx] = new_smiles[idx]
                # otherwise, if it hasn't improved but the retain count is below 3, we keep the base model and increment the counter
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
                raise MutaGenError(f"Failed to write temporary smiles file: {e}")

        final_df = pd.DataFrame({'Final SMILES Candidates': base_smiles, 'pIC50 Values': base_score})

        optima_df = pd.DataFrame({'Optima SMILES': optima_smiles, 'pIC50 Values': optima_scores})
        optima_df = optima_df.drop_duplicates()

        optimized_df = pd.DataFrame({'Target SMILES': target_smiles, 'pIC50 Values': target_scores})
        optimized_df = optimized_df.drop_duplicates()

        final_df.to_csv(Path(self.cfg['predictions']) / f'{self.mdl_nm}_final_mutant_compounds.csv')
        optima_df.to_csv(Path(self.cfg['predictions']) / f'{self.mdl_nm}_local_optima_compounds.csv')
        optimized_df.to_csv(Path(self.cfg['predictions']) / f'{self.mdl_nm}_optimized_compounds.csv')


    def random_mutation(self, smiles):
        """
        Randomly Mutates a SMILES
        \n Flow: input SMILES -> converted to mol item -> mol item is mutated -> converted back to SMILES and returned
        """
        # convert the smiles into a mol object which rdkit can then handle and mutate
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        mol_edit = Chem.RWMol(mol)
        mutated_mol = None

        atom_indices = list(range(mol.GetNumAtoms()))
        bond_num = mol.GetNumBonds()
        if not atom_indices:
            return Chem.MolToSmiles(mol)

        frags =[
            Chem.MolFromSmiles('C'),
            Chem.MolFromSmiles('CC'),
            Chem.MolFromSmiles('CN'),
            Chem.MolFromSmiles('O'),
            Chem.MolFromSmiles('N'),
            Chem.MolFromSmiles('F'),
            Chem.MolFromSmiles('CO'),
            Chem.MolFromSmiles('S'),

            # functional groups
            Chem.MolFromSmiles('C(=O)O'),
            Chem.MolFromSmiles('S(=O)(=O)N'),
            Chem.MolFromSmiles('C(=O)'),
            Chem.MolFromSmiles('C#N'),
            Chem.MolFromSmiles('C(F)(F)F'),
            Chem.MolFromSmiles('OC'),

            # More
            Chem.MolFromSmiles('Cl'),
            Chem.MolFromSmiles('Br'),
            Chem.MolFromSmiles('C(C)C'),
            Chem.MolFromSmiles('C(C)(C)C'),
        ]

        if len(smiles) >= 3:
            # default choices, normal molecule
            mutation_type = random.choice(["add_group", "replace", "remove"])
        elif len(atom_indices) <= 2 or bond_num == 0:
            # force additions for very small molecules or single atoms to protect them from being deleted
            mutation_type = "add_group"
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

            # enhanced fragment handling
            if "." in mutated_smiles:
                frags = mutated_smiles.split('.')
                # keep largest fragment
                largest_fragment = max(frags, key=len)

                frag_mol = Chem.MolFromSmiles(largest_fragment)
                if (len(largest_fragment) < 3 or  # reasonable complexity
                        frag_mol is None or  # make sure it's not a single atom or is minimal
                        frag_mol.GetNumAtoms() <= 1 or
                        frag_mol.GetNumBonds() == 0):
                    return smiles  # return original if the fragment did not pass
                else:
                    mutated_smiles = largest_fragment
                    mutated_mol = Chem.MolFromSmiles(mutated_smiles)

            # regular check
            if (mutated_mol is None or
                mutated_mol.GetNumAtoms() <= 2 or
                mutated_mol.GetNumBonds() == 0):
                return smiles  # return original if mutation did not pass

            return mutated_smiles

        except Exception as e:
            print(f"{smiles} failed to be sanitized, keeping original: {e}")
            return smiles   # return the original if sanitization fails


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
        Returns number of lipinski rules that were passed
        \n mol. weight, octanol water part., H-bond donors and acceptors
        """
        mol = Chem.MolFromSmiles(smiles)

        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Lipinski.NumHDonors(mol)
            h_acceptors = Lipinski.NumHAcceptors(mol)
        except Exception as e:
            raise MutaGenError(f"Error calculating Lipinski Descriptors: {e}")

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
