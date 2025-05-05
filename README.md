## NOCTURNAL

*Exploring the dark chemical space.*

- developed by: Elliot Chan
- contact: elliotchan120@gmail.com

## [1] Overview - for busy readers.

NOCTURNAL (v2.0.0) is a ChEMBL database navigation-aided interface for training ML models on drug-protein potency & compound molecular fingerprint analysis. Models can then be used to perform the following:

- Predict candidate compounds’ potencies (pIC50 values) against the respective target protein.
- Optimize drug candidate structures by being deployed in a molecular optimization algorithm system (class MutaGen) that stochastically explores chemical space aiming to generate improved drug candidate analogs. Heuristic techniques maintain oral bioavailability properties of all candidates produced.

All of this occurs within a modular, fault-tolerant architecture aimed at accelerating drug discovery workflows.

## [2] Project Flow Overview - How to use it!

If you’re not a super busy reader, let me take you through a test run of my project. Everything here is executable in 01_main.py, which is where I called on the classes during (lots of trial-and-error) testing. Usage of this project may need a bit of understanding so I hope this helps allow the reader to utilize it no matter the level they’re at. 

**[1] Data Scouting - Seeing what target to… target**

Firstly, a target protein has to be selected. I was interested in the Tau protein in Alzheimer’s at the time of making this project. So we’ll be using that as an example. 

‘target_input’ is the target protein you want to look at, ‘size’ is how many options you want to see.

```python
data_scout(target_input = “Tau”, size = 20)
```

data_scout will query the ChEMBL database for drug data on that target protein, and will output data as such:

> INDEX[25] CHEMBL1907600 | Homo sapiens | PROTEIN COMPLEX | 2447 IC50 entries
> 
> INDEX[12] CHEMBL2366565 | Sus scrofa | PROTEIN FAMILY | 753 IC50 entries
> 
> INDEX[13] CHEMBL2828 | Homo sapiens | SINGLE PROTEIN | 703 IC50 entries
> 
> INDEX[23] CHEMBL4036 | Homo sapiens | SINGLE PROTEIN | 475 IC50 entries


Data indices are automatically ranked according to number of IC50 entries, which we will later convert into pIC50, which is a metric for drug potency, I just pasted the first 4 results. Each entry contains a chemical compound that was tested against that target protein, and had its IC50 value recorded. With some more data processing that occurs in the next class, we can obtain their chemical formulas as a string called SMILES, which looks like this: “O=C1Cc2c([nH]c3ccc(Br)cc23)-c2cc(Br)ccc2N1” - CHEMBL98360

If you need more info:

IC50 = half-maximal inhibitory concentration. In other words: What concentration of ‘x’ do I need to inhibit a biological target ‘y’ by 50%?

pIC50 = -log_10(IC50) → a potency metric. Higher = more potent, note that IC50 is in Molarity (M)

**[2] Data Processing and Model Training**

Let’s say I wanted to use the first option, which is data index number ‘25’, I selected it since its got the greatest amount of IC50 entries and the data source is Homo sapiens | PROTEIN COMPLEX. Now I want to build a ML model using this data. I will first call on DataSeekProcess, pass on the target protein name and index, then say .run()

```python
DataSeekProcess(target_protein = “Tau”, selected_target_index = 25, fingerprint_setting “PubChem”).run()
```

“PubChem” refers to the specific settings that we use for molecular fingerprint generation. If you didn’t understand that, welcome to the club. Joking. Think of a molecular fingerprint as a way to represent chemical structures as 1’s and 0’s (that’s basically it). There’s various settings in the “PADEL SETTINGS” → “settings” section of the config file which, depending on the name chosen, will affect how each molecular substructure is encoded in the final dataframe which we build our machine learning models on. 

Once it’s done running, the “database” folder will have generated a bunch of files, which you don’t need to worry about for now. What DataSeekProcess has done is essentially created a DataFrame (CSV file) containing each compound’s molecular fingerprints and their potencies / pIC50 values.

When we call on class ModelBuilder, it will access our database folder and look at the file we just produced with DataSeekProcess. This is actually a good time to backtrack and note that I actually designed this class with modularity to select which machine learning algorithm you wish to choose. The current options are RandomForestRegressor, XGBoosting, and Stacking, which you can put in “ml_model_type” in the “MODEL TRAINING SETTINGS”. Here’s a brief rundown of what each one does. If you’re an expert on machine learning feel free to correct me on this.

- **RandomForestRegressor**: builds a lot of decision trees from random data subsets + features, then averages their predictions. It works well with “noisy” and non-linear data.
- **XGBoost / XGBRegressor**: builds trees sequentially, where each tree learns from the mistakes of the previous ones. It’s fast, accurate, and handles missing data.
- **Stacking**: trains multiple base models - I chose RandomForestRegressor, XGBRegressor, and Support Vector Regression. Each base model makes predictions while a “meta-learner” (Ridge Regression) looks over them and learns how to combine them. It leverages the strengths of multiple algorithms, and is usually more accurate than any single model.
    - This cranked my CPU temperature to 79 celsius though, so that may be a drawback if your operating system is on the low end.

That was a lot. Well now we can actually build the machine learning model! We don’t need to pass anything on to it other than the name we want to give our model.

```python
ModelBuilder(’test_model_1”).build()
```

After running this, we now have a folder named “test_model_1” in your “ml_models” folder as well as a few performance metrics about our model in the assessments folder (explained in the core features section of this readme). Inside this folder is the machine learning model saved as a .pkl file and a .txt file called “[name]_settings.txt”. We need both, as the settings file serves as a filter for future data to be passed through and be accessible by the ML model we made. 

We now have built a machine learning model that can predict ANY chemical compound’s pIC50 / potency against the “Tau” protein! Let’s see how we’re gonna do exactly that.

**[3] Running predictions!**

Remember at the start when we saw a SMILES? Every chemical compound has a representation as a SMILES. Let’s say you’re in a lab and they want to explore how compounds ‘x’, ‘y’, ‘z’, ‘x2’, ‘y2’, ‘z2’ will perform against a certain disease, whose target protein they’ve identified and entries already exist in ChEMBL… 

Well I’m not in a lab (yet), I’m in a basement, and I don’t have advanced chemical compounds from a lab, I have my old organic chemistry II notes from which I gathered a few random chemicals and put through an online “chemical name to SMILES converter”. I then put these SMILES into a file called “test_smile.smi” which we need to move to the “input_folder”. The file looks like this:

> filename: test_smile.smi
> 
> C1CCCCC1	Cycloh  exane
> 
> CCO	Eth   anol
>
> CC(=O)O	Acetic  acid
> 
> CCCCCC	Hexane
> 
> C1=CC=CC=C1	Ben  zene
> 
> CC(C)O	hljkjkh   iuhh iuh iu
> 
> C1CCOC1	   Tetrahydr  ofuran (THF)
> 
> C(C(=O)O)N	Glycine
> 
> CCN(CC)CC
> 
> O=C(C)Oc1ccccc1C(=O)O	Aspi   rin
> 

Wait, why are all the names (and not the SMILES…) messed up? My bad, my hands cramped from writing this extensive readme. Anyway, it doesn’t matter, because when we call on class RunModel and feed it this file, it will automatically validate it, and output its validated version. It will have a “VALID_” followed by your submitted file’s original name in the “input_folder”:

> filename: VALID_test_smile.smi
> 
> C1CCCCC1	Cyclohexane
> 
> CCO	Ethanol
> 
> CC(=O)O	Aceticacid
> 
> CCCCCC	Hexane
> 
> C1=CC=CC=C1	Benzene
> 
> CC(C)O	hljkjkhiuhhiuhiu
> 
> C1CCOC1	Tetrahydrofuran(THF)
> 
> C(C(=O)O)N	Glycine
> 
> CCN(CC)CC	compound_1
> 
> O=C(C)Oc1ccccc1C(=O)O	Aspirin
> 

Note how the second last compound didn’t have a name originally but now has the name “compound_1”.

This is all done automatically when you call on the class, so let’s look at what we need to pass on to it. Note how we didn’t say test_model_1.pkl or test_smile.smi, we just used the raw names with no file suffixes. 

```python
RunModel(model_name = “test_model_1”, input_smiles_filename = “test_smile”, fingerprint = “PubChem”).run_predictions()
```

What happens when we run this, is our “test_smile” is validated, then each molecule is converted into a molecular fingerprint with the “PubChem” setting and its all output into a csv file in “input_folder//input_fingerprint_data”. Next, our model reads that file, and predicts how potent each molecule is going to be against the target protein it was originally trained against: Tau. The predictions are output in a folder called “predictions” along with a bar chart visually representing the data. 

> Molecule ID                  SMILES                                  pIC50
> 
> 
> 
> | Cyclohexane | C1CCCCC1 | 4.909786429476615 |
> 
> | Ethanol | CCO | 4.823817551315969 |
> 
> | Aceticacid | CC(=O)O | 4.977223857317889 |
> 
> | Hexane | CCCCCC | 4.838476784385388 |
> 
> | Benzene | C1=CC=CC=C1 | 4.909786429476615 |
> 
> | hljkjkhiuhhiuhiu | CC(C)O | 4.9389590279949935 |
> 
> | Tetrahydrofuran(THF) | C1CCOC1 | 4.92096979543497 |
> 
> | Glycine | C(C(=O)O)N | 4.526465737739071 |
> 
> | compound_1 | CCN(CC)CC | 4.817367072558987 |
> 
> | Aspirin | O=C(C)Oc1ccccc1C(=O)O | 4.834832673200845 |

It’s very important to keep the fingerprint setting constant throughout. I’m going to add in automatic fingerprint detection later on so the user doesn’t have to bother with constantly saying they want the setting to be “PubChem” (remind me to remove this sentence after it’s done).

It should also be noted that pIC50 values of 6+ are generally strong, while 4- are pretty weak. This is just a sample of examples, but you can test it out using the actual database drugs, which can be found inside the database folder, in file “3_pIC50_dataclass.csv”. I just used these since they’re not super complicated / long.

Wow, in a couple lines of code (and a lot of reading, I’m sorry,) we managed to predict how effective these compounds would be against Tau proteins! If you think this is cool, you’re right, but I think we can go one step further.

**[4] Molecular Optimization: generating new chemical compounds!**

The crown jewel of this entire project. I named the class algorithm system “MutaGen” as it essentially “mutates” chemical compounds and “generates” new compound analogs, which will most likely have an increased pIC50 value by the end of the function. 

It should be noted that I have no clue if this exact algorithm may be novel, I developed it independently from 1-6am after a random epiphany and am unaware of any identical implementations in the field. I do intend to tweak it in the future with the purpose of optimized results. Let’s run the algorithm.

We call MutaGen like so:

```python
MutaGen(model_name = ‘test_model_1’, fingerprint_setting: ‘PubChem’).init_optimize()
```

Essentially this algorithm uses stochastic (randomness) functions to introduce random chemical mutations in the most potent compound calculated by the ML model we named. After each random mutation, the mutant is converted into a molecular fingerprint, and has its pIC50 predicted. The relevant parameters are in the config file → “candidates” and “iterations”. 

The default is 10 candidates and 100 iterations, meaning 10 of the starting compound will undergo 100 random mutations

I put a few other conditions in there to ensure that the pIC50 increases most of the time with each iteration, and that the drug remains as orally bioavailable as possible. 

This is what it looks like at the start in your text editor’s run window:

> Starting SMILES
> 
> ['CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O', 'CC(=O)O']
> 
> Iteration 0 / 100
> 
> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

This is what it should look like in the middle. 

> Iteration 61 / 100
> 
> [1, 1, 1, 2, 1, 2, 0, 3, 1, 1]
> 
> keeping original
> 
> C.N.O=CO.OS: 5.106306685986625 -- Retain_Count = 2
> 
> keeping original
> 
> NCC(CCC(F)SF)C(N)COCF.O: 5.584522383428918 -- Retain_Count = 2
> 
> keeping original
> 
> CN.Cl.O.OCCO.ON(S)S: 5.01309550301873 -- Retain_Count = 2
> 
> keeping original
> 
> C=O.N.NCCCO: 5.32517189253863 -- Retain_Count = 3
> 
> keeping original
> 
> CF.O.O.O=COO.OF: 5.109178672239924 -- Retain_Count = 2
> 
> keeping original
> 
> OOOF: 5.021848840128122 -- Retain_Count = 3
> 
> keeping original
> 
> Br.N.OCN(F)NO.S: 4.943618929310658 -- Retain_Count = 1
> 
> CCNC(=O)OC(O)CNC: 5.180781883373486 -- Retain_Count = 0
> 
> keeping original
> 
> [CCO.CO.NC](cco.o.co)(N)(O)CNO.O.O: 5.1246622653017635 -- Retain_Count = 2
> 
> keeping original
> 
> O=C(O)NOCO: 5.131364864065399 -- Retain_Count = 2
> 
- retain count represents the amount of times the compound failed to improve, once it gets past 3, we let it be ‘worse’ for a little to explore other optimization paths

The final results are output as a csv file:

> Optimized SMILES Candidate            pIC50 values
> 
> 
> 
> | C.O.O=CO.OS | 5.106306685986625 |
> 
> | --- | --- |
> 
> | CCCC(CO)C(N)COCF.F.O | 5.692708424736482 |
> 
> | C.CC(CO)CCN.Cl.Cl.N.S | 5.248891743061632 |
> 
> | CCN.CCNC(=O)O.N=O.NC=O | 5.363326743060715 |
> 
> | COC(=O)CF.NOO.OCS.S | 4.99415264727772 |
> 
> | O=C(O)N(F)S | 5.15914139974095 |
> 
> | FS.N.NN(N)F.NO | 4.890303990578768 |
> 
> | C.CCC(N)C(O)OC(=O)CCN.O.O | 5.737677841851358 |
> 
> | CCN.CCO.CO.CO.N.O.O=C(O)O.S.S.S | 5.1189746377138645 |
> 
> | CN.COCN.F.N.O | 4.346049170437902 |
> 

I know, not all of them are optimized, 2/10 have a lower pIC50 than the initial 4.9 something. However, the rest are greater, and if we had more candidates / more iterations / both, the chances of obtaining a more optimized compound is greater!

## **That marks the end :)**

I just took us through one run, from peering into a chemical database, to generating chemical compounds optimized towards a target disease protein of our choice! 

P.S. If there’s anything wrong with the project or any information that could be clarified, I would be very happy to learn as I’m fairly new to programming and have less than a month’s experience with machine learning.

## [3] Core Features - NOCTURNAL’s Architecture

This section goes deeper into the core algorithms and background processes that bring about the results it produces.

**[1] Config-driven architecture and automatic config key + folder validations**

- The 0_config.py file allows for customizability of a lot of core processes, like how many attempts one wishes to attempt at making molecular fingerprints as well as default parameters for training future machine learning models
- Furthermore, file b01_utility.py contains many custom error classes that help pinpoint the user towards the source of any mishaps during runs. E.g. ModelBuilderError, RunModelError .etc
- The validate_config() function in b01_utility is called upon every single class instantiation throughout the entire pipeline: it validates that all the keys in the config file, and all the required folders are present in their respective places. If those conditions are fulfilled, it loads the config file. Otherwise a custom “ConfigurationError” is raised.

**[2] Navigation-aided approach**

- data_scout()’s aim is to cut down on decision making by automatically sorting data indices by largest IC50 data entries to lowest, and it takes this one step further by outputting data quality information, such as where the data came from. This is relevant since its generally better to train models on larger amounts of higher quality data.

**[3] ModelBuilder’s modular feature organization**

- Class ModelBuilder allows one to choose what kind of model to build among the choices RandomForestRegressor, XGBoostRegressor, and Stacking RFR, XGBR, and SVR with Ridge as the meta-learner.
- RFR and XGBR go through hyperparameter optimization using GridSearchCV, while Stacking remains default (computationally expensive and time-comsuming on top of greater overfitting risk on smaller datasets. I might still add it in later on though).
- All go through sequential model evaluation: hold-out test set followed by k-fold cross validation. Model evaluation metrics such as R^2, RMSE, MAE on top of model performance and feature importance graphs are saved into the assessments folder.

**[4] MutaGen’s algorithm system (I built this all by myself by the way don’t be too mean)**

- Stochastic approach to generating varied compounds: the algorithm system utilizes a random_mutation() function to randomly introduce fragment addition, atom replacement and removal to simulate chemical space exploration
- Heuristic approach to validating compounds and breaking local optima: the new compound must fulfill two rules to pass on to the next iteration:
    1. It must have a pIC50 value greater than 0.05 (will make this configurable)
    2. It must fulfill minimum 2/4 Lipinski rules of oral bioavailability
    - If it fails to meet one of these, it does not make it through to the next iteration and the previous is kept. A “keep_counter” integer associated with each candidate compound is incremented by +1. When this number reaches 3, the rules to pass on change, allowing mutations resulting in negative pIC50 changes down to -0.5 and 0+ lipinski criteria fulfilled. The aim of this is to explore other chemical space orientations and escape plateaus in pIC50.
- Mutations are guided by a curated set of bioactive fragments instead of random atoms, improving the chemical realism of generated structures.
- Chemical Validity Filters: all mutations are first validated using valence checks and RDKit sanitizations. Fragment size filters are also applied to remove unstable or irrelevant candidates.
- Adaptive Logic to protect compounds: Short SMILES are protected from being destabilized further or eliminated by removal mutations, and hydrogen atoms are never used as connection points so valencies are constantly in check.
- P.S. this is definitely still a work in progress, I have a list of things I would definitely change to improve its performance that is already in the works.

## Future Improvements

- alter MutaGen logic to prevent candidates from being present in the final dataframe with a lower pIC50 value
- fingerprint settings in cfg file
- automatic fingerprint setting detection in RunModel and MutaGen
- logging instead of print statements (I got carried away with everything else)
- chemical space visualization
- drug candidate visualizations

## Prerequisites and Dependencies

- Python 3.7+
- Required packages:
    - requests
    - chembl_webresource_client
    - pandas
    - numpy
    - rdkit
    - padelpy -> also search online and download the PaDEL descriptor software, unzip the package and copy all the xml files to a new folder in your project directory
    - scikit-learn
    - xgboost
    - seaborn
    - matplotlib
    - pyyaml
