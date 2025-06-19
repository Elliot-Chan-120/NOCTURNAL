from a0_data_scout import data_scout
from a01_data_seek_process import DataSeekProcess
from a02_model_builder import ModelBuilder
from a03_run_model import RunModel
from a04_mutagen import MutaGen
from a05_csnodes import *
from a06_chemnet import ChemNet

def demo_datascout():
    data_scout("Tau", 20)

def demo_dataseekprocess():
    DataSeekProcess("Tau", 1, "PubChem").run()

def demo_modelbuilder():
    ModelBuilder('type_2_bot').build()

def demo_runmodel():
    RunModel("test_model_1", "test_smile").run_predictions()

def demo_optimizecompound():
    MutaGen('test_model_1').init_optimize()

def csn_data():
    csn_dataprocessor("test_model_1", "optimized")
    csn_dataprocessor("test_model_1", "optima")

def csn_network():
    ChemNet("test_model_1", "optimized").graph_data()
    ChemNet("test_model_1", "optima").graph_data()


csn_data()
csn_network()
