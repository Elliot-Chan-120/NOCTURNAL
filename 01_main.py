from a0_data_scout import data_scout
from a01_data_seek_process import DataSeekProcess
from a02_model_builder import ModelBuilder
from a03_run_model import RunModel
from a04_mutagen import MutaGen

def demo_datascout():
    data_scout("Tau", 20)

def demo_dataseekprocess():
    DataSeekProcess("Tau", 25, "PubChem").run()

def demo_modelbuilder():
    ModelBuilder('test_model_1').build()

def demo_runmodel():
    RunModel("test_model_1", "test_smile", "PubChem").run_predictions()

def demo_optimizecompound():
    MutaGen('test_model_1', 'PubChem').init_optimize()

demo_optimizecompound()
