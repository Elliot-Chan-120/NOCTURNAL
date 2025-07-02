from rdkit import Chem


class DataSeekProcessError(Exception):
    """Raised when there is an error with DataSeekProcess"""
    pass

class ModelBuilderError(Exception):
    """Raised when there is an error with ModelBuilder"""
    pass

class RunModelError(Exception):
    """Raised when there is an error with RunModel"""
    pass

class MutaGenError(Exception):
    """Raised when there is an error with MutaGen"""
    pass

class CSNDataError(Exception):
    """Raised when there is an error with data processing for the chemical space network"""
    pass

class ChemNetError(Exception):
    """Raised when there is an error with the chemical network visualization module"""

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
        'predictions', 'database', 'input_folder', 'assessments', 'model_folder', 'input_fp_folder',
        'padel_xmls', 'network_folder',
        'rawdf_1', 'labeldf_2', 'pIC50df_3', 'smile_data', 'fp_output', 'fingerprintdf_4',
        'ml_model_type',
        'try_limit', 'settings', 'train_test_split', 'cross_validate', 'grid_search', 'use_best_params',
        'random_forest_params',
        'candidates', 'iterations', 'target_increase', 'error_threshold', 'success_threshold', 'retain_threshold',
        'data_scout_csv', 'auto_save_model', 'n_features',
        'colorscale', 'transparent_nodes', 'node_toggle', 'label_toggle', '2D_molecules', 'node_size', 'tanimoto_bias',
        'target_size'
    ]

    for key in required_keys:
        if key not in cfg:
            raise ConfigurationError(f"Missing Required Configuration Key: {key}")

    # 2. Now access this config and make sure all the necessary starting folders are there
    dir_keys = ['predictions', 'input_folder', 'assessments', 'model_folder',
                'database', 'padel_xmls', 'optimizer_database']

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


# MUTATION FRAGMENT LIBRARY
nonaroma_frags = [
    # simple chains / single atoms
    Chem.MolFromSmiles('C'),   # methyl
    Chem.MolFromSmiles('CC'),   # ethyl
    Chem.MolFromSmiles('CN'),   # methylamino
    Chem.MolFromSmiles('O'),   # hydroxy
    Chem.MolFromSmiles('N'),   # amino
    Chem.MolFromSmiles('F'),   # fluoro
    Chem.MolFromSmiles('CO'),   # methoxy
    Chem.MolFromSmiles('S'),   # thiol

    # functional groups
    Chem.MolFromSmiles('C(=O)O'),   # carboxyl
    Chem.MolFromSmiles('S(=O)(=O)N'),   # sulfonamide
    Chem.MolFromSmiles('C(=O)'),   # formyl
    Chem.MolFromSmiles('C#N'),   # cyano
    Chem.MolFromSmiles('C(F)(F)F'),   # trifluoromethyl
    Chem.MolFromSmiles('OC'),   # methoxy

    # More
    Chem.MolFromSmiles('Cl'),   # chloro
    Chem.MolFromSmiles('Br'),   # bromo
    Chem.MolFromSmiles('C(C)C'),   # isopropyl
    Chem.MolFromSmiles('C(C)(C)C'),   # tert-Butyl
]

aromatic_frags = [
    # Simple carbon strings
    Chem.MolFromSmiles('C'),  # methyl
    Chem.MolFromSmiles('CC'),  # ethyl
    Chem.MolFromSmiles('CCC'),  # propyl
    Chem.MolFromSmiles('C(C)C'),  # isopropyl
    Chem.MolFromSmiles('CCCC'),  # butyl
    Chem.MolFromSmiles('C(C)(C)C'),  # tert-Butyl

    # Halogens
    Chem.MolFromSmiles('F'),  # fluoro
    Chem.MolFromSmiles('Cl'),  # chloro
    Chem.MolFromSmiles('Br'),  # bromo
    Chem.MolFromSmiles('I'),  # iodo

    # Oxygen groups
    Chem.MolFromSmiles('O'),  # hydroxy
    Chem.MolFromSmiles('OC'),  # methoxy
    Chem.MolFromSmiles('OCC'),  # ethoxy
    Chem.MolFromSmiles('OCCC'),  # propoxy
    Chem.MolFromSmiles('OC(C)C'),  # isopropoxy

    # Nitrogen groups
    Chem.MolFromSmiles('N'),  # amino
    Chem.MolFromSmiles('NC'),  # methylamino
    Chem.MolFromSmiles('N(C)C'),  # dimethylamino
    Chem.MolFromSmiles('NCC'),  # ethylamino
    Chem.MolFromSmiles('N(CC)CC'),  # diethylamino

    # Carbonyls
    Chem.MolFromSmiles('C(=O)C'),  # acetyl
    Chem.MolFromSmiles('C(=O)CC'),  # propionyl
    Chem.MolFromSmiles('C(=O)O'),  # carboxyl
    Chem.MolFromSmiles('C(=O)OC'),  # methyl ester
    Chem.MolFromSmiles('C(=O)N'),  # carboxamide
    Chem.MolFromSmiles('C(=O)NC'),  # N-Methylcarboxamide

    # Sulfur groups
    Chem.MolFromSmiles('S'),  # thiol
    Chem.MolFromSmiles('SC'),  # methylthio
    Chem.MolFromSmiles('S(=O)C'),  # methylsulfinyl
    Chem.MolFromSmiles('S(=O)(=O)C'),  # methylsulfonyl
    Chem.MolFromSmiles('S(=O)(=O)N'),  # sulfonamide

    # Special groups
    Chem.MolFromSmiles('C(F)(F)F'),  # trifluoromethyl
    Chem.MolFromSmiles('OC(F)(F)F'),  # trifluoromethoxy
    Chem.MolFromSmiles('C#N'),  # cyano
    Chem.MolFromSmiles('C[N+](=O)[O-]'),  # nitro

    # Aromatic rings
    Chem.MolFromSmiles('c1ccccc1'),  # phenyl
    Chem.MolFromSmiles('c1ccc2ccccc2c1'),  # naphthyl
    Chem.MolFromSmiles('c1ccncc1'),  # pyridyl
    Chem.MolFromSmiles('c1cccnc1'),  # pyridyl
    Chem.MolFromSmiles('c1ccoc1'),  # furanyl
    Chem.MolFromSmiles('c1ccsc1'),  # thienyl

    # Heterocyclic fragments
    Chem.MolFromSmiles('C1CCNCC1'),  # piperidyl
    Chem.MolFromSmiles('C1COCCN1'),  # morpholinyl
    Chem.MolFromSmiles('C1CCN(C)CC1'),  # N-Methylpiperidyl
]
