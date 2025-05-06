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

## Future Improvements
alter MutaGen logic to prevent candidates from being present in the final dataframe with a lower pIC50 value
fingerprint settings in cfg file
automatic fingerprint setting detection in RunModel and MutaGen
logging instead of print statements (I got carried away with everything else)
chemical space visualization
drug candidate visualizations


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

