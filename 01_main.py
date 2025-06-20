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
    DataSeekProcess("Tau", 20, "PubChem").run()

def demo_modelbuilder():
    ModelBuilder('test_model_1').build()

def demo_runmodel():
    RunModel("test_model_1", "test_smile").run_predictions()

def demo_optimizecompound():
    MutaGen('test_model_1').init_optimize()

def csn_data(model_name):
    csn_dataprocessor(model_name, "optimized")
    csn_dataprocessor(model_name, "optima")

def csn_network(model_name):
    ChemNet(model_name, "optimized").graph_data()
    ChemNet(model_name, "optima").graph_data()


csn_data("test_model_1")
csn_network("test_model_1")
