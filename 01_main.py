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
    DataSeekProcess("Tau", 25, "PubChem").run()

def demo_modelbuilder():
    ModelBuilder('test_model_1').build()

def demo_runmodel():
    RunModel("test_model_1", "benchmark_smile").run_predictions()

def demo_optimizecompound():
    MutaGen('test_model_1').init_optimize()

def csn_data(model_name):
    if __name__ == '__main__':
        csn_dataprocessor(model_name, "optimized", filter_strategy='performance')
        csn_dataprocessor(model_name, "optima", filter_strategy='balanced')

def csn_network(model_name, weight_method):
    ChemNet(model_name, "optimized", weight_method, filter_strategy='performance').graph_data()
    ChemNet(model_name, "optima", weight_method, filter_strategy='balanced').graph_data()

csn_network('test_model_1', 'hybrid')
