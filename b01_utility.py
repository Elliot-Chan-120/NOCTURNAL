class DataSeekProcessError(Exception):
    """Raised when there is an error with DataSeekProcess"""
    pass

class ModelBuilderError(Exception):
    """Raised when there is an error with ModelBuilder"""
    pass

class RunModelError(Exception):
    """Raised when there is an error with RunModel"""
    pass

class OptimizeError(Exception):
    """Raised when there is an error with OptimizeMolecule"""
    pass

class ConfigurationError(Exception):
    """Raised when there is an error with config"""
    pass

class ChEMBLAPIError(Exception):
    """Raised when there's an error with ChEMBL API"""
    pass

class DataProcessingError(Exception):
    """Raised when there's an error with data processing"""
    pass

class PaDELProcessError(Exception):
    """Raised when there's an error with PaDEL fingerprint outputs"""
    pass


def validate_config():
    """Validates all keys in the config file and folder locations -> returns config"""
    from pathlib import Path
    import yaml

    # 1. Verify config file exists and load if true
    required_cfg = '0_config.yaml'
    if not Path(required_cfg).exists():
        raise ConfigurationError(f"Config file missing: {required_cfg}")

    with open('0_config.yaml', 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config: {e}")

    # validate config file for all of its critical keys
    required_keys = [
        'predictions', 'bioactivity_folder', 'input_folder', 'assessments', 'model_folder', 'input_fp_folder',
        'padel_xmls',
        'rawdf_1', 'labeldf_2', 'pIC50df_3', 'smile_data', 'fp_output', 'fingerprintdf_4',
        'ml_model_type',
        'try_limit', 'settings', 'train_test_split', 'cross_validate', 'grid_search', 'use_best_params',
        'random_forest_params',
        'candidates', 'iterations', 'target_increase', 'error_threshold', 'success_threshold', 'retain_threshold',
        'data_scout_csv', 'auto_save_model', 'n_features'
    ]

    for key in required_keys:
        if key not in cfg:
            raise ConfigurationError(f"Missing Required Configuration Key: {key}")

    # 2. Now access this config and make sure all the necessary files are there
    dir_keys = ['predictions', 'input_folder', 'assessments',
                'model_folder', 'bioactivity_folder', 'padel_xmls']

    required_dirs = [cfg[k] for k in dir_keys]

    for folder in required_dirs:
        if not Path(folder).exists():
            raise ConfigurationError(f"Critical Folders Missing: {folder}")

    return cfg


def get_fingerprint(config, model_name):
    """Scans chosen ml model's settings and returns fingerprint key"""
    from pathlib import Path

    try:
        with open(Path(config['model_folder']) / model_name / f'{model_name}_settings.txt', 'r') as handle:
            column = handle.readline()
            name = column.strip().split(',')
            item = name[0].lower()

            for key in config['settings']:
                if key.lower() in item.lower():
                    return key
    except Exception as e:
        raise ConfigurationError(f"Error occurred while loading fingerprint setting: {e}")

