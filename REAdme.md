
This reposity contains the code used in the Bank of England Staff Working Paper 984 _An interpretable machine learning workflow with an application to economic forecasting_ by Marcus Buckmann and Andreas Joseph.

In the paper, we propose a generic workﬂow for the use of machine learning models to inform decision making and to communicate modelling results with stakeholders. It
involves three steps: (1) a comparative model evaluation, (2) a feature importance analysis and (3) statistical inference based on Shapley value decompositions. The paper 
demonstrates each step by forecasting changes in US unemployment one year ahead using [FRED-MD](https://research.stlouisfed.org/wp/more/2015-012), a US data base of monthly macroeconomic measurements.

The code published here is not intended as a stand-alone package. It can be used to reproduce the results of the paper. But as the porposed workflow is quite general, parts of it may be transfered to other applications. No warranty is given. Please consult the licence file.

Should you have any questions or spot a bug in the code, please send an email to marcus.buckmann@bankofengland.co.uk or raise an issue within the repository.


The code has been developed under Python 3.9.7 Anaconda distribution. The file requirements/requirements.txt specifies the Anaconda environment in which the experiments were run.



# Structure of the code

We use ```Python``` to train and test the machine learning models and compute the measures of variable importance and use ```R``` to produce the Figures and Tables presented in the paper.

## Modelling


The script ```experiment.py``` is the main script. It reads the data, and calls the ```run_experiments``` function, which, in a loop conducts all the experiments.
The parameters of the experiments are read from a yaml file that is created with the script ```setup_epxeriments.py```. In that script, the user can specify which prediction methods to test and can set parameters of the experimental set-up, such as the degree of winsorising, the type cross-valdiation used for hyperparameter optimisation, or the training sample size. The user can either set these parameters to a single value (e.g. method =  "Forest") or to a list of several values. In the latter case, all parameter combinations will be enumerated and the ```run_experiments``` will produce results for each of the experiments.

The ```run_experiments``` function computes the predictions of the model as well as the variable importance measures (Shapely values, permutation importance). For each individual experiment, output files with the preditions, and variable importance measures are saved on the hard drive in the __results__ folder. The name of these files are hash keys that are based on the parameters of the experimental setup (see ```setup_experiments.py```.

























# Disclaimer
This package is an outcome of a research project. All errors are those of the authors. All views expressed are personal views, not those of any employer.
