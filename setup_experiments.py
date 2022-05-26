import yaml

experiment_setup = {
    "name_ID": "main",
    "bootstrap_hyper": False,  # let the hyperparmaeter search be based on a bootstrap sample. Then the different rnadom iterations will lead to different hyperpparameters even on determinis models such as SVM
    "bootstrap_proportion": 1, # Size of the bootstrapped training samples relative to the training sample size. E.g. 1 means that we sample as many observations with replacement as there are in the training set.
    "change_predictor": 3, # time difference to compute percentage changes, and differences of the predictors
    "change_target": 12,# time difference to compute percentage changes, and differences of the target (and lagged target as a predictor)
    "cv_mode": 3, # after how many time steps should the hyperparameter search be updated: e.g 5 ~ every 5th time step. The time step is determined by step_size
    "cv_repeat": 1, # how often is the cross-validation procedure for the hyperparameter tuning repeated 
    "counter": [0], # how often experiment is repeated with different random seed. List with Ids, e.g. list(range(10)). Counters parameter, as all other parameters also influences the seed
    "error_metric": "absolute_error", # only relevant for grid search loss metric and Forest loss metric
    "features": ["key"], # specifies on which set of featuresthe prediction models are train (either: "key" - see config.py, "all", or "pca")
    "gap_size_kfold_block": 12, # size of the gap for the blocked kfold hyper parameter estimation
    "hyper_type": ["kfold_block_gap"], # type of cross-validation used for the hyperparameter search
    "random_search_iterations": 100, # the number of parameter settings that are sampled in the random search of the hyperparameter optimisation
    "seed": [None], #if none no random seed is set for the hyperparameter tune, if true a seed is set, which affects (1) the random hyperparamter combinations tested and (2) the assignment to the folds
    "lag": 12, # lag predictive variables
    "lag_response_as_predictor": ["response"], # can  be "response" (change is the same as response), "predictor" (change is the same as predictor), "no" (not used)
    "norm_var": True, # whether the predictors are normalised when training and testing the prediction models
    "n_folds": 5, # number of folds used in hyperparameter cross-validation
    "n_boot": [30], # number of bootstrap iterations of the training set on which the prediction models are trained
    "n_boot_importance": [30], # number of bootstrap iterations of the training set on which the importance measures (Shapely values, ...) are computed
    "method": ["Ridge", "LightGBM"], # models used (see utils.sk_model_dict)
    "permutation_rep": 250, # number of iterations when computing permutation importance
    "pca_max_components": 2, # if features = "PCA", this specifies on the number of components the ML models are trained on
    "step_size": 12, # after how many months the model should be updated
    "target": "UNRATE",# name of the target variable in the data set
    "window_size": 10000, # maximum size of the training set, if number is smaller, a rolling window is used for training to adapt to sturctural shifts. With very high numbers, all past data is used for training
    "winsorize": [.01] # level fo winsorisation
    }
with open("setup_files/main.yaml", 'w') as outfile:
    yaml.dump(experiment_setup, outfile, default_flow_style=False)
