from __main__ import *

# CV VALUES for grids search
nn0, nnf   = 5, 5  # parameters for NN, specifies number of neurons per layer


"""
look-up dictionary for the machine learning model calls
requires package imports according to the import+[ackages script]
"""
sk_model_dict = {
                'SVM': "skl_svm.SVR",
                'NN' : "skl_nn.MLPRegressor", 
                'Forest': "skl_ens.RandomForestRegressor",        
                'Forest_fast': "skl_ens.RandomForestRegressor",   
                'Forest_rich': "skl_ens.RandomForestRegressor",                                       
                "GP": "GaussianProcessRegressor",
                'Enet': "skl_lin.ElasticNetCV",
                "LightGBM":  "lightgbm.LGBMRegressor",
                "Ridge": "skl_lin.Ridge",
                "Lasso": "skl_lin.Lasso",
                "OLS": "skl_lin.LinearRegression"
                 }


# it is a function, because some hyperparameters depend on an input
"""
Default hyperparameter settings for the machine learning (ML) methods

:param do_boot: whether ML model model is bootstrapped
:param exp: dictionary containing experimental configuration
:return: dict of parameter settings
"""

def hyper_default(do_boot, exp): 
    dict_out = {
        "OLS": {"fit_intercept": True},    
        "Lasso": {"random_state": exp["seed"]},
        "Ridge": {"random_state": exp["seed"]},
        "NN": {
                "solver": "lbfgs",
                "max_iter": 2000,
                "verbose": False,
                "random_state": exp["seed"]
                },       
        "Forest": {
            'n_estimators': int(np.where(do_boot, 25, 500)),
            'criterion': str(np.where(exp["error_metric"] == "absolute_error", "mae", "mse")),
            'n_jobs': 1,
            "random_state": exp["seed"]
        },
        "LightGBM": {
            "boosting_type": "gbdt", 
            "n_jobs": 1,
            "random_state": exp["seed"]
            }
    
    }
    return(dict_out)           




"""
specification of the hyperparamter grid for the machine learning methods
"""
hyper_grid  = {
            'NN': {'hidden_layer_sizes': [(n)   for n in range(nn0,nn0*nnf+1,nn0)]+\
                                               [(n,n)   for n in range(nn0,nn0*nnf+1,nn0)]+\
                                               [(n,n,n) for n in range(nn0,nn0*nnf+1,nn0)],
                         'alpha':              10.**np.arange(-5, 3),
                         'activation':         ['relu','tanh'],
                    },
            'Forest': {'max_depth':          [2, 3, 4, 5, 6, 8, 10, 20, 30],
                       'max_features':       [1, 3, 5, 7, 9, 11, 15, 30, 50]
                        },

            'SVM':        {'C':                  2. ** np.linspace(1., 5., 10),
                         'gamma':              2. ** np.linspace(-7.,-1, 10),
                         'epsilon':            [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "LightGBM": { # extended to make it work on single features
                    'subsample': [0.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
                    'reg_lambda': [0, 1e-1, 1,10, 20, 50, 100], 
                    'reg_alpha': [0, 1e-1, 1, 2, 7, 10, 50, 100],
                    'num_leaves': [2,3,4, 5,10,20,40, 70, 100], 
                    'n_estimators': [1,3, 5, 10, 20, 30, 40, 50, 75, 100], # , 200, 300, 500, 1000, 2500], we extended that but it was not necearry (on key and all features)
                    'max_depth': [1, 2, 3, 5, 8, 15], #, 30, 50], we extended that but it was not necearry (on key and all features)
                    'colsample_bytree': [.2, .3, .4, .5, .6, .7, 1]},

            "Lasso": { # try whether this works
                "alpha": list(np.logspace(-5, 4, num = 100))
            },
            "Ridge": { # try whether this works
                "alpha": list(np.logspace(-5, 4, num = 100))
            }
}
              
                     

""""
Construct the model string that is used to 
:param method: name of the ML model
:param do_boot: whether ML model model is bootstrapped
:param exp: dictionary containing experimental configuration
:hyper_input: hyperparameters that have previously been determined by a cross-validated search
:return: string that will be used to create a machine learning model object.
"""
def make_model_str(method, do_boot, exp, hyper_input = {}):
    
    model_str = sk_model_dict[method]
    try:
        hyper_list = hyper_grid[method].copy()
    except:
        hyper_list = {}
    try:
        defaults = hyper_default(do_boot, exp)[method].copy()
        for key in defaults.keys():
            if isinstance(defaults[key], str):
                defaults[key] = "'" + defaults[key] + "'"
    except: 
        defaults = {}

    all_params = defaults
    all_params.update(hyper_list)
    all_params.update(hyper_input)
    # construct parameter string
    params_str   = ''        
    for _, p in enumerate(all_params):

        add_parameter = all_params[p]
        # if isinstance(add_parameter, str):
        #     add_parameter = "'" + add_parameter + "'"


        params_str += p + '=' + str(add_parameter) + ','
    # join model and parameters    
    model_str += '({0})'.format(params_str[:-1]) # remove last comma

    return model_str

"""evaluates model string
:param method: name of the ML model
:param do_boot: whether ML model model is bootstrapped
:param exp: dictionary containing experimental configuration
:hyper_input: dict of hyperparameters that have previously been determined by a cross-validated search
:return: returnss initialised model object
"""
def model_selection(method, do_boot, exp, hyper_input={}):
    return(eval(make_model_str(method, do_boot, exp, hyper_input = hyper_input)))





"""
conducts hyperparamter search for machine learning models
:param df: data set on which the grid search is conducted
:param exp: dictionary containing experimental configuration
:param features: list containing the names of the features used for modelling
:param pca: boolean whether a pca should be used to condense the feature space before the training the model
:return: returns tuple of hyperparameters and time (in seconds) used to calibrate them
"""
def cv_hyper(df, exp, features, pca = False):
    
    if exp["bootstrap_hyper"]:
        df = df.sample(frac=.8, replace=False).sort_index()
    if exp["error_metric"] == "absolute_error":
        scoring = "neg_mean_absolute_error"
    if exp["error_metric"] == "squared_error":
        scoring = "neg_mean_squared_error"
    
    t0 = time.time()
    if exp["norm_var"]:
        df[features] = normalise(df[features])[0]
    
    # set type of cross-valdiation procedure
    if exp["hyper_type"] == "kfold":      
        folds = skl_slct.RepeatedKFold(n_splits = exp["n_folds"], n_repeats = exp["cv_repeat"], random_state = exp["seed"])    
    if exp["hyper_type"] == "kfold_block":     # see: http://www.zhengwenjie.net/tscv/
        folds = tscv.GapKFold(n_splits = exp["n_folds"])
    if exp["hyper_type"] == "kfold_block_gap": # see: http://www.zhengwenjie.net/tscv/
        gap = exp["gap_size_kfold_block"]
        if gap is None:
            gap = exp["lag"]
        folds = tscv.GapKFold(n_splits = exp["n_folds"], gap_before = gap, gap_after = gap) #  + exp["change_predictor"]
    
    # set default parameters
    # note sklearn expect defualt parameters to be lists rather than int, float, or str
    try:
        default_params = hyper_default(do_boot = False, exp = exp)[exp["method"]]
        default_param_list = {key: [default_params[key]] for key in default_params.keys()} # default parameters need to be in a list
    except:
        default_param_list = {}

    param_list = dict(hyper_grid[exp["method"]], **default_param_list)
        
    model = eval(sk_model_dict[exp["method"]])()
    if pca:
        model = Pipeline([('pca', PCA(n_components = exp["pca_max_components"])), ('model', model)])
        for key in list(param_list.keys()):
            param_list["model__" + key] = param_list.pop(key) # parameters in pipeline object automatically get a prefix, which we need to add to the parameter dictionary

    cv = skl_slct.RandomizedSearchCV(estimator = model, n_iter = exp["random_search_iterations"],
                                    param_distributions = param_list,
                                    scoring = scoring, cv = folds, n_jobs = config.n_cpus, verbose = 0, random_state = exp["seed"])
    cv.fit(df[features], df["target"])
    
    if pca: # remove prefix that was added due to the pipeline
        for key in list(cv.best_params_.keys()):
            cv.best_params_[re.sub("model__", "", key)] = cv.best_params_.pop(key)

    for k in cv.best_params_.keys():
        if type(cv.best_params_[k]) == str: # if parameter is a string, we need to add extra quotation marks
            cv.best_params_[k] = '"' + cv.best_params_[k] + '"'        
    hyper = cv.best_params_
    return (hyper, time.time() - t0)
   


"""
function training and testing a selected prediction model
:param df_train: pd.DataFrame training set (pandas data)
:param df_test: pd.DataFrame test set 
:param features: list of feature names used for prediction
:param exp: dictionary containing experimental configuration
:param hyper: dict of hyperparameters that have previously been determined by a cross-validated search
:param save_models: boolean whether models should be added to the output of the function 
:param models_input list of previously trained models (several when using bootstrapping). Default None. When not none, models is not retrained but only tested
:param pca: boolean whether a pca should be used to condense the feature space before the training the model
:return: dict containing predictions on test set, training set, and out-of-bag observations of the training 
"""

def model_tester(df_train, df_test, features, exp, hyper=None, save_models = True, models_input = None, pca = False):
    
    train_time = []
    n_train = len(df_train)
    n_boot = exp["n_boot"]
   
    if  df_test is None: # if we only want to do bootstrapping we should allow for test sets that are "None"
        n_test = 0
        do_test = False
    else: 
        n_test = len(df_test)
        do_test = True
        
    # empty fields for bootstrapped model output
    test_pred = np.empty(shape=(n_boot, n_test)) * np.nan
    train_fit = np.empty(shape=(n_boot, n_train)) * np.nan 
    train_boot_pred = train_fit.copy() 
    boot_train_ixs = []   
    
    if models_input is None:
        models = [] #  and bootstrap models
        train = True
    else:
        models = models_input            
        train = False
    # loop over bootstrapped samples
    seed = exp["seed"]
    for t in range(n_boot):
        if not seed is None:
            seed = seed + 1
        train_boot, test_boot, test_index_bool, train_index = sample_bootstraps(df_train, bootstrap_proportion = exp["bootstrap_proportion"], do_boot = t > 0, seed = seed) # random split
        x_train_boot, y_train_boot = train_boot[features].values, train_boot["target"].values
        
        boot_train_ixs.append(train_index)
        # get values
        x_test_boot = test_boot[features].values
        if do_test:
            x_test = df_test[features].values
            if exp["norm_var"]:
                x_train_boot, x_test, x_test_boot = normalise(x_train_boot, x_test, x_test_boot)
        else:
            if exp["norm_var"]:
                x_train_boot, x_test_boot = normalise(x_train_boot, x_test_boot)

        
        if exp["method"] == "AR1":  # AR1 model
            n_lags = 1
            t0 = time.time()
            if train:
                model = AR(y_train_boot) 
                model = model.fit(maxlag = n_lags, ic = None) 
            else:
                model = models[t]
            train_time.append(time.time() - t0)
            
            fit_y = model.predict(start = 0 + n_lags, end = len(y_train_boot) -1)
            fit_y = np.append(np.zeros(n_lags) * np.nan, fit_y)
            if do_test:
                pred_y = model.predict(start = len(y_train_boot) , end = len(y_train_boot) + len(x_test) - 1) 
          

        elif exp["method"] == "AR_auto": # AR model with 1-12 lags, lags are determined by AIC
            t0 = time.time()
            if train:
                model = AR(y_train_boot) 
                model = model.fit(ic = "aic", maxlags = 12) 
            else:
                model = models[t]

            train_time.append(time.time() - t0)
            n_lags = len(model.params) - 1
            fit_y = model.predict(start = 0 + n_lags, end = len(y_train_boot) -1)
            fit_y = np.append(np.zeros(n_lags) * np.nan, fit_y)
            if do_test:
                pred_y = model.predict(start = len(y_train_boot), end = len(y_train_boot) + len(x_test) - 1)
        
        # machine learning models
        else:

            t0 = time.time()
            if train:
                model = model_selection(method = exp["method"], do_boot = t > 0, exp = exp, hyper_input = hyper)
                if pca:
                    model = Pipeline([('pca', PCA(n_components = exp["pca_max_components"])), ('model', model)])
                model = model.fit(x_train_boot, y_train_boot) # model fit
            else:
                model = models[t]

            train_time.append(time.time() - t0)
            fit_y = model.predict(x_train_boot)

            if do_test:
                pred_y = model.predict(x_test)
            if t > 0:
                boot_y = model.predict(x_test_boot)
                train_boot_pred[t, test_index_bool] = boot_y
        
        # save outputs for all models
        train_fit[t, np.unique(train_index)] = fit_y[np.unique(train_index, return_index = True)[1]] 
        if do_test:
            test_pred[t, :] = pred_y
        
        if save_models and train:
            models.append(model) 
    # package output
    out_dict = {"test_pred": test_pred, 
                "time_train": sum(train_time),
                "train_fit": train_fit,
                "train_boot_pred": train_boot_pred, 
                "boot_train_ix": boot_train_ixs,
                "models": models
                }    
    return out_dict





""" This is the main experimental function that iterates through the experiments,, trains and 
    tests the machine learning models and computes Shapley values
:param data_in: pd.DataFrame that contains the dataF set (including response variable and predictors)
:param experiments: pd.DataFrame of experiments, which swe iterate through
:param transformations_in: dictionary listing the transformations for each of the variables (e.g. log differences)
:param overwrite: boolean; if true previous results (of the same experiment) are 
                            overwritten, if false, experiment is skipped
:param importance_tasks: list of tasks (str) to compute variable importance (Shapley values, Permutation importance ,...)
:param test_time: list of tasks ("forecast", "out-of-bag") that determine how we assess the performance 
                and variable importance. "forecast" means proper out-of-sample forecasting. 
                "out-of-bag" means training the models on a bootstrapped sample and testing them
                on the out-of-bag sample
:param out_of_bag_dates: list of the dates up to which we train the boostrapped models and evaluate their out-of-bag predictions


"""


def run_experiments(data_in, 
                            experiments,
                            transformations_in,
                            overwrite = False,
                            importance_tasks = [],
                            test_time = ["forecast"],
                            out_of_bag_dates = ['1999-12-01', '2008-08-01', '2019-11-01']
                            ):
           
    # iterate through all specified experiments
    for _, exp in experiments.iterrows():
        
        transformations = transformations_in.copy()
        data = data_in.copy()
        data["target"] = data[exp["target"]].copy()
        hyper = {}
        if exp["features"] == "none":
            features = []
            if exp["lag_response_as_predictor"] in ["predictor", "response"]:
                features = features + [exp["target"]]

        if exp["features"] in ["all", "pca_by_group", "pca"]:
            features = list(transformations.keys())
        if exp["features"] == "key":
            features = config.features_key.copy()
            if exp["lag_response_as_predictor"] in ["predictor", "response"]:
                features = features + [exp["target"]]
            data["YIELDC"] = data["GS10"] - data["TB3MS"]
            transformations["YIELDC"] = 2
        if exp["features"] == "lag_dependent":            
            features = []
        
        # Check whether we already obtained results
        # currently only relevant for local execution: see also below the run_experiment boolean for the cloud
        
        
        if isinstance(test_time, list):
            file_check = "results_" + "_".join(test_time)
        else:
            file_check = "results_" + test_time
        if (not overwrite) and file_exists(exp["output_folder"] + file_check + str(exp["hash"]) + ".pickle"):
            print("Experiment skipped. Already conducted")
            continue
        if  file_exists(exp["output_folder"] + '_placeholder_'+ file_check + str(exp["hash"]) + ".txt"):
            print("Experiment skipped. Experiment is currently being conducted")
            continue
        try:
            # create a placeholder file that indicates that the experiment (with the specific hash is in progress. In this way
            # this experiment is not conducted by two different threads, when we use parallel computing
            file = open(exp["output_folder"] + '_placeholder_' + file_check + str(exp["hash"]) + ".txt", "w") 
            file.write("experiment is conducted") 
            file.close() 
        except:
            pass
        
        transformations["target"] = str(transformations[exp["target"]]) + "-" + str(exp["change_target"])            
        transformations = {i:transformations[i] for i in ["target"] + features} 
        
        # Code transformation span of predictors. 
        # Also, the parameter lag_response_as_predictor determines whether the transformation of the response is that of 
        # the predictor or that of the response.
        for key in features:
            if key == exp["target"]: # just try this
                if exp["lag_response_as_predictor"]  == "predictor":
                    lag_response_as_predictor = exp["change_predictor"]
                if exp["lag_response_as_predictor"]  == "response":
                    lag_response_as_predictor = exp["change_target"]
                transformations[key] = str(transformations[key]) + "-" + str(lag_response_as_predictor) # this is the lagged target as a predictor 
            else:    
                transformations[key] = str(transformations[key]) + "-" + str(exp["change_predictor"])
    

        df_use = transform_data(data.copy(),
                                index = config.time_var,
                                start_ix = config.start_time,
                                end_ix = config.end_time,
                                transformations = transformations,
                                target = "target",
                                lag_indicators = exp["lag"])
        
        
    
        # index on row of the data set where we start the test set
        time_first_test = int(np.where(df_use.index.values == np.datetime64(config.periods["all"][0]))[0]) 
        time_first_test = time_first_test - 12 * 6 # start a few years early to make sure we actual have a valid test set according 
                                                   # to the period data after controlling the lag between trianing and test et
        
        # determine the end time steps of the training sets that we will iterate through when recursively training the models
        # the step size parameter determines how often the models are updated
        end_training_times = df_use.index[time_first_test -1:][np.arange(0, len(df_use) - time_first_test, exp["step_size"])]    
        end_training_times = end_training_times[end_training_times <= df_use.index.max()]

        if not "forecast" in test_time: # only do out-of-bag estimation (no forecasting)
            end_training_times = np.array(out_of_bag_dates, dtype=np.datetime64) # boot dates show the last training date
        else:         
            if "out-of-bag" in test_time:
                if not np.all(np.isin(np.array(out_of_bag_dates, dtype=np.datetime64), end_training_times)): # if boot_dates are not covered by the training sequence
                    raise ValueError("The selected bootstrapping dates (parameter out_of_bag_dates) are not in the sequence of test sets considered."\
                        "Change out_of_bag_dates accordingly or do not use forecasting and boostrap simultaneously as prediction tasks."
                    )

        model_dict = {}
        output_dict = {"exp": exp, "results": {}}        
        
        # iterate through all time steps at which we train a new model
        for t, end_time_train in enumerate(end_training_times):
           
            features_use = list(df_use.columns[1:])
            end_time_train_ix = int(np.where(end_time_train == df_use.index)[0])
            # create training and test sets
            if exp["window_size"] is None: 
                start_time_ix = 0
            else:
                training_size = np.min([end_time_train_ix, exp["window_size"]])
                start_time_ix = end_time_train_ix - training_size + 1
                
            start_time_train = df_use.index[start_time_ix]
            df_train = df_use[(df_use.index >= start_time_train) & (df_use.index <= end_time_train)]
            try:
                df_test = df_use[(df_use.index > end_time_train)]
            except: # test set might be completely empty
                df_test = None
            if df_test.shape[0] == 0:
                df_test = None

            if exp["winsorize"] > 0.0: # winsorise features
                features_winsorise = features_use.copy()
                if "USREC" in features_winsorise: features_winsorise.remove("USREC") # lagged dependent variable should not be winsorised

                lower_wins = df_train[features_winsorise].quantile(exp["winsorize"])
                upper_wins = df_train[features_winsorise].quantile(1 - exp["winsorize"])

                df_train[features_winsorise] = df_train[features_winsorise].clip(lower = lower_wins, upper =upper_wins, axis = 1).copy()
                if not df_test is None:
                    df_test[features_winsorise] = df_test[features_winsorise].clip(lower = lower_wins, upper =upper_wins, axis = 1).copy()
            
            # do hyperparameter search
            do_hyper = t == 0 # at the first time step, we need to find hyperparameters
            do_hyper = do_hyper or (t % exp["cv_mode"] == 0) # and we do it at every "cv_mode" time step. E.g. if model is trained once a year (step_size = 12)
                    # cv_mode =3, means that the hpyerparameters are updated every 3 years.
            do_hyper = do_hyper or (not "forecast" in test_time)  # when we do not do forecasting but only out-of-bag performance analysis, 
                # we also need to tune the hyperparameters
            do_hyper = do_hyper and (exp["method"] in hyper_grid.keys()) # we only do the hyperparameter search for models that have hyperparameters

            if do_hyper:
                hyper, _ = cv_hyper(df_train.copy(), exp, features_use, pca = exp["features"] == "pca")                            

            # train the model and obtain the predictions
            perf_results = model_tester(df_train, df_test, features = features_use,
                                    exp = exp, save_models = (len(importance_tasks) > 0),
                                    hyper = hyper, pca = exp["features"] == "pca")

            #### MODEL SAVING ####
            predictions = copy.deepcopy(perf_results)
            del(predictions["models"])
            predictions["test_set"] = df_test
            predictions["training_set"] = df_train
            predictions["end_train_time"] = end_time_train
            predictions["hyper_params"] = hyper
            output_dict["results"][str(t)] = predictions.copy()
            output_dict["features"] = features_use
            model_dict[str(t)] = perf_results['models']
    

        output_dict["exp"] = exp
        save_name  = exp["output_folder"] + file_check + str(exp["hash"])
        write_pickle(output_dict, save_name)
            

        # obtain variable importance estimates
        for importance_task in importance_tasks:
                if ("out-of-bag" in test_time): 
                    
                    # compute variable importance for the out-of-bag predictions                    
                    end_train_times_keys = [str(int(np.where(np.datetime64(k) <= end_training_times)[0][[0]])) for k in out_of_bag_dates]
                    for r in range(len(out_of_bag_dates)): # iterature through the dates at which we bootstrapped our models
                        importance_bootstrap = compute_importance_summary_bootstrap(output_dict, model_dict,
                                                                                    boot_use = exp["n_boot_importance"],
                                                                                    ix_use = end_train_times_keys[r], 
                                                                                    method = importance_task)
                        save_name  = exp["output_folder"] + "importance_oob_" + importance_task + "_" + "period_" + out_of_bag_dates[r] +  "_" + str(exp["hash"]) 
                        write_pickle(importance_bootstrap, save_name)

                if "forecast" in test_time: 
                    # compute variable importance for the forecasting predictions                    
                    lag = exp["lag"]
                    if not isinstance(lag, int):
                        lag = max(lag)
                    importance_forecast = compute_importance_summary(output_dict, model_dict,
                                                                    boot_use = exp["n_boot_importance"],
                                                                    forecast_lag = lag,
                                                                    method = importance_task)
                    save_name  = exp["output_folder"] + 'importance_forecast_' + importance_task + "_" + str(exp["hash"]) 
                    write_pickle(importance_forecast, save_name)

        # delete placeholder file
        try:
            os.remove(exp["output_folder"] + '_placeholder_' + file_check + str(exp["hash"]) + ".txt")
        except:
            pass

"""
This function collects the predictions of a single recursive forecasting experiment. Each of these experiment
 produces a sequence of models each making a prediction for the next few time steps before a more recently tained model
 is used. This function collects all these predictions of the different models into a a single series of predictions.
:param input_all: nested dictionary containing all the results of the forecasting experiment 
:param boot_use: int specifying how many bootstrap iterations (if model has been bootstrapped) will be used for prediction.
    If None all available bootstrapped iterations will be used
:param forecast_lag: The length of the period the model has to wait until it makes predictions. This makes sense from a policy perspective.
                     We want tp make predictions not one time point ahead (a month) but longer
:param forecast_lag: int The length of the period the model has to wait until it makes predictions. This makes sense from a policy perspective.
                     We want tp make predictions not one time point ahead (a month) but longer
:return: dictionary with observed and predicted response and date across the whole prediction period
"""
def aggregate_predictions(input_all, boot_use = None, forecast_lag = 12):

    
    input_res = input_all["results"]
    exp = input_all["exp"]

    boot_max = input_res[list(input_res.keys())[0]]["test_pred"].shape[0]
    if (boot_use is None) or (boot_use > boot_max):
        boot_use = boot_max

    boot_ix = list(range(boot_use))

    boot_prediction = None
    
    forecast_predictions = np.array([])    
    forecast_true_y = np.array([]) # can be longer than actual data set, if horizon is long than gap
    forecast_dates = np.array([],  dtype='datetime64[ns]')

    for i in input_res.keys():
        
        if ("train_boot_pred" in input_res[i]) and (not input_res[i]["train_boot_pred"] is None):
            boot_prediction = np.nanmean(input_res[i]["train_boot_pred"][boot_ix, :], axis = 0) # this is updated over time, but we are interested in the latest bootstrap sample
    
        horizon = exp["step_size"]
        # we have to skip the first observations if we have a yearly lag otherwise we would not have a true yearly forecast. 
        skip_first_obs = forecast_lag - 1
        forecast_period = np.arange(horizon) + skip_first_obs
        forecast_period = forecast_period[forecast_period < len(input_res[i]["test_pred"][0])]
        if len(forecast_period) == 0:
            continue
            
        mean_pred = np.nanmean(input_res[i]["test_pred"][np.ix_(boot_ix, forecast_period)] , axis = 0)
        true_y = input_res[i]["test_set"]["target"].values[forecast_period]
        date = input_res[i]["test_set"].index.values[forecast_period]
        forecast_predictions = np.append(forecast_predictions, np.array(mean_pred))
        forecast_true_y = np.append(forecast_true_y, np.array(true_y))
        forecast_dates = np.append(forecast_dates, np.array(date))

    output = {        
        "forecast_dates": np.array(forecast_dates),
        "mean_forecast": np.array(forecast_predictions),
        "boot_dates": input_res[i]["training_set"].index.values,
        "boot_prediction": boot_prediction,
        "boot_true": input_res[i]["training_set"]["target"].values,
        "forecast_true": np.array(forecast_true_y),
        "boot_use": boot_use
    }
    return (output)



"""
This function merges the prediction results with the experimental parameters. 
It creates an output table for each period that is considered
:param predictions: dict containing the predictions and observed values for each experiment. 
    The hash of the experiment is the key of the dict
:param table: pd.DataFrame containing the experimental parameters of all experiments
:param periods: dict containing the period(s) for which the output table is prepared
:return: dict of pd.DataFrames containing the predictions and true values + experimental paramters for selected period
"""
def prediction_table(predictions, table, periods):
        table_descriptors = table.copy()
        output = {}
        dates_forecasted = [list(predictions[key]["forecast_dates"]) for key in predictions.keys()]
        dates_forecasted = list(chain(*dates_forecasted))
        dates_forecasted = pd.Series(dates_forecasted).value_counts()
        dates_forecasted = dates_forecasted.index.sort_values()
        dates_forecasted = dates_forecasted.sort_values()
        
        for period_key, p  in periods.items():
            dates_period = dates_forecasted[(dates_forecasted >= p[0]) & (dates_forecasted <= p[1])]            
            out_singles = []
            for key in predictions.keys():
                    forecasts = predictions[key]["mean_forecast"][np.isin(predictions[key]["forecast_dates"], dates_period)]
                    observed = predictions[key]["forecast_true"][np.isin(predictions[key]["forecast_dates"], dates_period)]
                    dates = predictions[key]["forecast_dates"][np.isin(predictions[key]["forecast_dates"], dates_period)]
                    joined =  {"pred": forecasts, "true":observed, "date": dates}
                    joined.update(table_descriptors.loc[key,:])
                    joined = pd.DataFrame(joined)
                    out_singles.append(joined)              
            output["pred_" + period_key] = pd.concat(out_singles)
            
        return output
            
