from a0_data_scout import data_scout
from a01_data_seek_process import DataSeekProcess
from a02_model_builder import ModelBuilder
from a03_run_model import RunModel
from a04_mutagen import MutaGen

def demo_datascout():
    data_scout("Diabetes", 20)

def demo_dataseekprocess():
    DataSeekProcess("Diabetes", 1, "PubChem").run()

def demo_modelbuilder():
    ModelBuilder('type_2_bot').build()

def demo_runmodel():
    RunModel("type_2_bot", "test_smile").run_predictions()

def demo_optimizecompound():
    MutaGen('test_model_1').init_optimize()

demo_optimizecompound()