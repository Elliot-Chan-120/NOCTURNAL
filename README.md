Essentially a Customizable Machine Learning Pipeline, this project is capable of generating unique machine learning models from a set of data acquired through ChEMBL, a chemical database containing information on bioactive molecules with drug-like properties.
It then allows for prediction of any chemical compound's potency against a specified target protein via any of the produced machine learning models. The entire process of this and classes are explained below.

DataScout: does a quick search and returns the number of data entries per index produced by the chembl search. Returns the largest 20 sorted greatest to smallest.
- Useful since the model will work better with more data. Choosing an index/target protein with 4 will most likely not produce good results if the data does make it pass the data processing classes.
- the example i used was "Tau" and index: 42 since there were many data points and allowed for a higher-accuracy machine learning model to be produced. 

DataSeekProcess class: seeks data on the specified target protein and its index, then processes bioactivity data from the database.
Initially searching for a target protein and fetches relevant bioactivity data and passes it down a pipeline of data processing and 'cleaning' functions until a molecular fingerprint file is obtained through Padel (PadelPy). 
It is then combined with pIC50 values to generate the processed dataframe (.csv) for the next class.

ModelBuilder: builds regression machine learning model based on the previously parsed data.
Performs a little more cleaning (removing low variance data) and generates a machine learning model that is immediately assessed and returns an R^2 value as an accuracy metric.
Prompts the user with y/n if they are satisfied with the accuracy of the produced model or if they want to try and get a better one. 'y' to save model and its settings with the pickle library (as pkl), 'n' to reject, necessitating a re-run of the program.
Various machine learning models can be produced, the class just needs to be called again with a different desired name .e.g 'Elliot_Tau_idx12' or 'This_model_will_be_accurate'. 
Produces a scatterplot graph with a regression line with the predicted vs experimental pIC50 values as a visual representation of the model's accuracy.

ModelRun: runs predictions on user-provided SMILES-formatted compounds using the model, outputting a .csv containing the molecular ID or Name, canonical SMILES, and pIC50 values.
Accepts a Simplified Molecular Input Line Entry System, or SMILES file containing one or more compounds and predicts each of their pIC50 (Drug Potency) against the initially targetted protein in DataSeekProcess.
With a specified saved model, each compound will have their pIC50 values predicted and a .csv file containing their ID/Names, SMILES, pIC50 values will be put into an output file named by the user.
Includes a SMILES validate function in case the file provided by the user is incorrectly/inconsistently spaced (but does require at least one space in between the compound and ID), or does not label the compound -> outputs a validated smiles file (.txt).

REQUIREMENTS (pip install these):
- chembl_webresource_client
- pandas
- numpy
- seaborn
- padelpy -> also search online and download the PaDEL descriptor software, unzip the package and copy all the xml files to a new folder in your project directory
- scikit-learn
- lazypredict

Future improvements:
- Finish off the compare_model(self) function -> compares model performance against other models using lazyregression
- More comms and docstrings
- Data Visualizations
- maybe a GUI
