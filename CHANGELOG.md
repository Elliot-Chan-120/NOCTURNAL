NOCTURNAL v2.0.0 (May 4th 2025)

Added the complete codebase for NOCTURNAL: a machine learning pipeline designed for computational drug discovery.

## Key Features:

- New Class: MutaGen. Enables one to use a molecular optimization algorithm, guided by pIC50 increases. Designed to explore the nearby chemical space and generate new chemical analogs with greater pIC50 values while maintaining Lipinski rule compliance
    - *Author’s note*: This algorithm may be novel, I developed it independently and am unaware of any identical implementations in the field. I do intend to tweak it in the future.
- ChEMBL Database scouting for improved, calculated target-protein data selection
- Molecular fingerprinting and feature engineering
- Multiple ML model implementations: RandomForest, XGBoost, Stacking - RF, XGBR, SVR w/ Ridge regression as meta-learner
- Hyperparameter optimization via GridSearchCV
- Sequential model evaluation: hold-out test set followed by k-fold cross-validation
- Comprehensive model evaluation metrics (R², RMSE, MAE)
- Visualization tools for model assessment and feature importances

## Technical Improvements:

- Robust error handling with custom exception classes for pipeline-critical events: PaDEL processes and molecular fingerprint data integrity, cross validation, model training, saving, loading and predictions .etc
- Configuration-driven architecture and approach using YAML for flexible settings
- Graceful degradation for external dependencies
- File I/O safety checks throughout pipeline

This update represents a majoroverhaul to the previous RandomForest-only prototype. It introduces a wider range of models, hyperparameter tuning, rigorous evaluation, and an original chemical optimization algorithm (MutaGen), all within a modular, fault-tolerant architecture aimed at accelerating drug discovery workflows.


NOCTURNAL v2.1.0 May 5th 2025
- Added more config variables to further control MutaGen
	- 'target_increase' -> how much of an improvement in pIC50 we are aiming for
	- 'error_threshold' -> how much error we are willing to accept once an optima has been hit
	- 'success_threshold' -> the minimum increase in pIC50 for a compound to be deemed a success and make it to the next iteration
	- 'retain_threshold' -> how many times the molecule can fail to improve before being considered an optima 

- Altered MutaGen's logic -> now keeps track of 3 types of compounds during the optimization process and outputs each in their respective csv files in folder 'predictions'
	- "[model_name]_local_optima_compounds.csv": compounds that are local optima (failed to be improved x times)
	- "[model_name]_final_mutant_compounds.csv": final candidate compounds after all iterations have been completed
	- "[model_name]_optimized_compounds.csv": successful compounds that have been optimized to meet or exceed the pIC50 target improvements

- Added a function in b01_utility: get_fingerprint(config, model_name)
	- allows classes to automatically obtain the fingerprint configuration the model was trained on. Repeatedly specifying the fingerprint method across all classes is no longer necessary. 

## Future improvements: 
- Chemical space analysis + visualizations
- Logging instead of print statements
- Safety analysis
- Optimized + Optima Compound Visualization

NOCTURNAL v2.2.0 June 11th 2025
Improvements to the MutaGen algorithm

- modified adaptive escape mechanism: success criteria now becomes more strict the more time the molecule fails to increase in potency -> produces fewer candidates but properly ensures that they are of higher quality.

- added more mutation fragment possibilities: added some functional groups, carbon chains and Cl + Br

- enhanced fragment handling: in the event of a fragmented molecule, the largest fragment alone is selected for the next round of predictions. This also clashed with a chemical space visualization module I was working on, as Tanimoto similarity calculations involved some processes that required intact molecules that were not overly simple (needed a minimum atom # and bond #).

- fixed / added addition step of optimized molecules to next iteration: the previous version 'quit' exploring once an optimized molecule was found by not letting it in to the next iteration. So the algorithm was basically limiting itself more up until now.


NOCTURNAL v2.3.0 June 19 2025
ADDED CHEMICAL SPACE VISUALIZATION MODULE

The methodology to generating the nodes and portions of the chemical space visualization code were inspired and derived from work by Vincent F. Scalfani (see NOTICE file).

## Chemical Space Visualization Module Features:
- introduced two new files containing toolsets containing specialized toolsets for visualizing the molecules MutaGen produces in chemical space:
	- a05_csnodes contains utility functions and the CSNodes class which function together in order to generate Tanimoto similarity data using rdkit's fingerprinting as well as node data containing SMILES and pIC50 values
	- a06_chemnet contains the ChemNet class which takes both the Tanimoto similarity and SMILES: pIC50 data in order to generate an interactive Chemical Space Network Graph
- molecules' potencies are shown by their highlight colour (colorscheme is configurable)
- molecules' 2D chemical structures are displayed directly on the graph 
Interactive features:
	- hover text shows pIC50, % rank (of pIC50) and the chemical string (SMILES)
	- plotly allows for zoom/pan functionality
Adaptive layout:
	- adaptive network density and 2D molecular image sizing to subset size to retain as much visual clarity as possible
- thorough error handling (I hope)
- graph cosmetic customizability from config file (can toggle the presence / absence of nodes, 2D images)


NOCTURNAL v2.4.0 - June 22 2025
Aromatic structure compatibility / optimization capability
Enhanced MutaGen by allowing it to work with aromatic structures with the following features:
- Moving from hardcoded fragments to imported fragment lists in the b01_utility file
- Intelligent fragment selection based on atom aromaticity: aromatic / non-aromatic structure-compatible fragment lists
- Kekulization and aromaticity management prior to molecular combinations
- Added a safe_selection() method with error handling during add mutations - prevents 'bad' additions with empty fragments



NOCTURNAL v2.5.0 - July 2 2025
Bottleneck solutions in a05_csnodes.py
a05_csnodes.py now allows for 3 intelligent sampling modes for MutaGen results that contain an extensive amount of molecules
- 'balanced': evenly samples across potency quartiles
- 'performance': selects the top compounds by pIC50
- 'mcs_optimized': prefilters by overall Tanimoto similarity and selects for the top 'n' to accelerate MCS calculations
	- NOTE: mcs_optimized samples may still require a significant amount of time to process since it requires all overall Tanimoto calculations prior to sorting and cutting down time on more intricate MCS calculations

Surrounding changes:
- Adjusted ChemNet, utility and config to allow for intelligent sampling functionality and keeping track of what sampling strategies were used on which dataset
- Reorganized README.md to accommodate these changes



Future Improvements:
- Logging functionality
- Continued enhancements towards MutaGen's algorithm.
- Chemical space network statistics
- Displaying optima and optimized compounds together


