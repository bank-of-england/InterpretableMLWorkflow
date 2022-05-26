
This reposity contains the code used in the Bank of England Staff Working Paper 984 _An interpretable machine learning workflow with an application to economic forecasting_ by Marcus Buckmann and Andreas Joseph.

In the paper, we propose a generic workﬂow for the use of machine learning models to inform decision making and to communicate modelling results with stakeholders. It
involves three steps: (1) a comparative model evaluation, (2) a feature importance analysis and (3) statistical inference based on Shapley value decompositions. The paper 
demonstrates each step by forecasting changes in US unemployment one year ahead using [FRED-MD](https://research.stlouisfed.org/wp/more/2015-012), a US data base of monthly macroeconomic measurements.

The code published here is not intended as a stand-alone package. It can be used to reproduce the results of the paper. But as the porposed workflow is quite general, parts of it may be transfered to other applications. No warranty is given. Please consult the licence file.

Should you have any questions or spot a bug in the code, please send an email to marcus.buckmann@bankofengland.co.uk or raise an issue within the repository.


The code has been developed under Python 3.9.7 Anaconda distribution. The file requirements/requirements.txt specifies the Anaconda environment in which the experiments were run. 



# Structure of the code

We use ```Python``` to train and test the machine learning models and compute the measures of variable importance and use ```R``` to produce the Figures and Tables presented in the paper. Below, provide a hgh level description how to run the experiments and analyse the results. The ```scr``` folder contains helper functions and  configurations. 

 
- ```utils.py``` contains general helper functions, e.g. reading and wirting data, or normalising and trasforming variables
- ```utils_predict``` contains functions used to train and test the prediction models, e.g. it specifies the hyperparameter grids of the machine learning models
- ```utils_importance``` contains the functions to estimate variable importance.
- ```uitls.R``` helper functions used when analysing the results in ```R```.
- ```config.py``` contains some configuration parameters that are constant across all experiments.
- ```import_packages``` imports all Python packages used.

## Modelling


The script ```experiment.py``` is the main script. It reads the data, and calls the ```run_experiments``` function, which, in a loop conducts all the experiments.
The parameters of the experiments are read from a yaml file (in folder ```setup_files```) that is created with the script ```setup_epxeriments.py```. In that script, the user can specify which prediction methods to test and can set parameters of the experimental set-up, such as the degree of winsorising, the type cross-valdiation used for hyperparameter optimisation, or the training sample size. The user can either set these parameters to a single value (e.g. method =  "Forest") or to a list of several values. In the latter case, all parameter combinations will be enumerated and the ```run_experiments``` will produce results for each of the experiments.

The ```run_experiments``` function computes the predictions of the model as well as the variable importance measures (Shapely values, permutation importance). For each individual experiment, output files with the preditions, and variable importance measures are saved as __pickle__ files on the hard drive in the __results__ folder. The name of these files are hash keys that are based on the parameters of the experimental setup (see ```setup_experiments.py```.

## Aggregating the results 

After running the individual experiments, their results can be aggregated using the ```collect...``` scripts. 

The script ```collect_performance.py``` appends the prediction reuslts of the individual experiments and save them in a __.csv__ in the folder __results/aggregated__.
Similarly, the scripts ```collect_importance_forecast.py``` aggregated the variable importance results into a single __.csv__ file for the Shapely vlaues and permutation importance measures in the forecasting experiments. In the paper we also estimate Shaeply values using the out-of-bag apporach. This means, we train the models on a bootstrapped sample of the data and estimate the Shapley vlaueson those observatios not in the boostrapped sample (i.e. the out-of-bag observations). The script ```collect_importance_out_of_bag.py``` collects these results.


## Shapley regression 

The script ```shapley_regression.py``` reads the file containint the Shapley values of the forecasting experiments (that was compiled by ```collect_importance_forecast.py```) and computes the Shapley regression ([see Joseph, 2019](https://aps.arxiv.org/abs/1903.04209v1)). 



## Analysing the results

The script ```error_analysis.R``` reads the aggregated prediction results copmutes the prediction errors and plots the time series.

The script ```shapley_analysis``` reads the aggregated Shapley vlaues permutation importance values and produces the main variable improtance figures shown in the paper.










# Disclaimer
This package is an outcome of a research project. All errors are those of the authors. All views expressed are personal views, not those of any employer.
