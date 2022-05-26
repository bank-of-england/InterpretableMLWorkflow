from __main__ import *
import os
import pickle
import errno
from statsmodels.tsa.seasonal import seasonal_decompose
import tempfile
import pandas as pdf
import hashlib
import itertools
import re


"""
checks whether file exists locally
:param file_name: str file name
:return: boolean that is true if file exists
"""
def file_exists(file_name):
    return(os.path.exists(file_name))
    
""""
reads pickle file from local drive
:param file_name: str the name of the file
:return: the loaded data
"""
def read_pickle(file_name):
    with open(file_name + ".pickle", "rb") as f:
        return pickle.load(f)

""""
write pickle file to local drive
:param file: file that is saved
:param file_name: str the name of the file
"""
def write_pickle(file, file_name):

    folder = os.path.dirname(file_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(file_name + ".pickle", "wb") as f:
        return pickle.dump(file, f, protocol=4)


        
"""
Normalise data to have a mean of 0 and standard deviation of 1
:param train: tpd.DataFrame input data sets that is used as reference when normalising the data
:param *args: additional data sets that will be normalised with respect to the train data.
:return: tuple of normalised data sets
"""
def normalise(train, *args):
    means = train.mean(0)
    sds = train.std(0)
    train = (train - means) / sds
    out = [train]
    for arg in args:
        out.append((arg - means) / sds)
    return(tuple(out))

"""
Creates a list by enumerating all combinations of the experimental parameters
:param exp_dict: dictionary of the experimental parameters
:return: returns the list of experimental configurations
"""
def model_val_params(exp_dict):
    # use model_spec_dict from function above (chosen in config file)    
    param_names = list(exp_dict.keys())
    params = list(exp_dict.values())
  
    # create dictionary of different model specifications given model_spec
    param_dict = dict(zip(param_names, params))
      
    single = []
    single_names = []
    comb = []
    comb_names = []
                    
    for k, v in param_dict.items():                
        # if a parameter (e.g. ML models to test) has moer than one parameter value, calculate all possible combinations
        if isinstance(v, list):
            comb.append(v)  
            comb_names.append(k)
        else:
            single.append(v)   
            single_names.append(k)                
    combinations = list(itertools.product(*comb))              
      
    # combine parameters with single and multiple values )
    final_comb = []        
    single_tuple = tuple(single) 
    for item in combinations:
        # add fixed parameters to every combination of varying parameters
        merge = item + single_tuple
        final_comb.append(merge)
            
    param_name_order = comb_names + single_names    
        
    # create list of dictionaries for each specification
    param_dict_list = []        
    for params in final_comb:
        values = [x for x in params]
        temp_dict = dict(zip(param_name_order, values)) 
        param_dict_list.append(temp_dict)
        
    return param_dict_list


"""
Creates a data frame listing all experiments by enurating all combinations of the experimental parameters
:param setup_file dictionary that contains all the relevant experimental parameters
"""
def list_experiments(setup_file):
    experiment_specs = setup_file.copy()
    # create dataset    
    experiments = model_val_params(experiment_specs)
    df_spec = pd.DataFrame(columns = ["hash"] + list(experiment_specs.keys()))    
    # add a unique hash to the parameter combination of the experiment
    for exp in experiments:
        spec_list = [exp[p] for p in experiment_specs.keys()]
        hash_name = ",".join(map(str, spec_list))
        hash_name = int(hashlib.sha256(hash_name.encode('utf-8')).hexdigest(), 16) % 10**14
        df_spec.loc[len(df_spec)] = [hash_name] + spec_list
    return(df_spec)


"""
Some experimental combination are effectively duplicates and can be removed.
This fucntion does that. For example, using different cross-validation proceudres for
hyperpater tuning is irrelevant for models without hyperparamters
:param experiments pd.DataFrame contianing all the experimentss
:return: pd.DAtaFrame of experiments removing the duplicates
"""
def remove_duplicates(experiments):
    experiments.loc[~np.isin(experiments["features"].values, ["pca", "pca_by_group"]), ["pca_max_components"]] = np.nan
    experiments.loc[experiments["method"].values == "OLS", ["cv_mode", "cv_repeat", "cv_type", "random_search_iterations"]] = np.nan
    ix_dup = experiments.astype(str).drop(columns = ["hash"]).duplicated()
    experiments = experiments.loc[~ix_dup,:].reset_index(drop = True)
    return(experiments)


"""
compute prediction errors
:param y_true: np.array (vector) of observed response
:param y_pred: np.array (vceotr) of predicted response
:param normalise: booleean whether the error should be normalised by 
    a simple baseline (mean prediction)
:return: different error metrics
"""
def compute_errors(y_true, y_pred, normalise = False):
    abs_deviance = np.abs(y_true - y_pred)
    abs_error = np.nanmean(abs_deviance)
    rmse_error = math.sqrt(np.nanmean((y_true - y_pred)**2))
    
    abs_mean_deviance = np.nanmean(np.abs(y_pred - np.nanmean(y_pred)))
    if normalise: # standardise by the error that the mean prediction would make
        naive_errors = compute_errors(y_true, np.tile(y_true.mean(), len(y_true)), standardise = False)
        abs_error = abs_error/naive_errors["abs_error"]
        rmse_error = rmse_error/naive_errors["rmse_error"]
    
    cor = np.nan
    if isinstance(y_true, (collections.abc.Sequence, np.ndarray)):
        a = np.ma.masked_invalid(y_true)
        b = np.ma.masked_invalid(y_pred)
        msk = (~a.mask & ~b.mask)
        if (y_true[msk].std() > 0) and (y_pred[msk].std() > 0):
            cor = np.corrcoef(y_true[msk], y_pred[msk])[0,1]
    return({
            "n": len(abs_deviance),
            "abs_error": abs_error, 
            "rmse_error": rmse_error,
            "correlation": cor,
            "abs_mean_deviance": abs_mean_deviance})



"""
transform variables to make them stationary or to compute lags
:param data: pd.Series containing the series to which the transformation is applied
:param trafo: string describing the transformation. E.g. 1-4, 
    where the first character describes the type of the transformation (see McCracken & Ng (2015)) 
    and the second parameter describes the horizon (in months) of the transformation
:return: pd.Series of the transformed input series
"""
def variable_transformer(data, trafo):
  
    # use number codes from the McCracken Paper describing the FRED MD data base
    tf = trafo.split('-')
    try:
        h = int(tf[1])
        lag = np.array(pd.Series(data).shift(h))
    except:
        pass    
    
    if tf[0] == '1': # no transformation
        return data

    elif tf[0] == '2': # first difference
        return (data - lag)

    elif tf[0] == '3': # second difference
        return variable_transformer(data, "2-" + str(h)) - variable_transformer(lag, "2-" + str(h))

    elif tf[0] == '4': # log 
        return (np.log(data)) 
    elif tf[0] == '5': # first log difference                
        return (np.log(data) - np.log(lag))
    elif tf[0] == '6': # second log difference
        return variable_transformer(data, "5-" + str(h)) - variable_transformer(lag, "5-" + str(h))
    elif tf[0] == '7': # delta percentage change
        return variable_transformer(data, "pc-" + str(h)) - variable_transformer(lag, "pc-" + str(h))
    elif tf[0] == 'lag':
        return(lag)
    elif tf[0] == 'pc': # percentage change
        return((data - lag) / lag)

    else:
        raise ValueError("Invalid transformation value.")


       
"""
Transforms the variables of the whole data set. By calling the variable_transformers

:param data_input pd.DataFrame input data
:param index: str name of the column that contains the date columns
:param start_ix: np.datetime that sets the start date, earlier dates are deleted
:param end_ix: np.datetime that sets the end date, later dates are deleted
:param transformations: dict containing the transformations (values) for the variables to be transformed (key). 
    Only those variables in the dict will be transformed    
:param target: str name of the target variable. In contrast to other variables, it will not be lagged
:param lag_indicators: int by how much the predictors are lagged
:param drop_na: boolean whether observations with missing values sohuld be dropped
:return: pd.DataFrame containing the transformed data



"""
def transform_data(data_input, index, start_ix, end_ix, transformations, target, lag_indicators = 0, drop_na = True):
    # data is a pandas data frame    
    data = data_input.copy()
    data.set_index(index, inplace=True)

    # data column transformations
    data_new = pd.DataFrame(columns=[index])
    data_new[index] = data.index
    data_new.set_index(index, inplace=True)
    # transform variables

    
    for key in transformations.keys():
        data_new[key] = variable_transformer(data[key].values, transformations[key]) # transformation (e.g. percentage change)
        if key != target: 
            if isinstance(lag_indicators, int): # single lag
                data_new[key] = variable_transformer(data_new[key].values, "lag-" +str(lag_indicators)) # add lag for the predicitve variables
            else: # multiple lags
                for lag in lag_indicators:
                    data_new[key + "_lag_" + str(lag)] = variable_transformer(data_new[key].values, "lag-" +str(lag)).copy() # add lag for the predicitve variables
                data_new = data_new.drop(columns = [key])

    if start_ix is None:
        start_ix = 0
    else:
        start_ix = list(data_new.index).index(start_ix)
    if end_ix is None:
        end_ix = len(data.index)
    else:
        end_ix = list(data_new.index).index(end_ix)

    data_new = data_new.iloc[start_ix:end_ix + 1,:]
   
    if drop_na:
        data_new = data_new.dropna()
    return(data_new)


"""
Samples with replacement from an input data set
:param data_input: pd.DataFrame input data set
:param bootstrap_proportion: float indicating the size of the bootstrap sample relative to the input data. 
    A value of 1 means the bootstrapped data is as big as the input data.
:param do_boot: boolean. If False, no sampling is applied and the input data set is returned
:param seed: int random seed
:return: a tuple of four objects is returned: 
        (1) the bootrapped data set, 
        (2) the out-of-bag data frame containing those observations not in the boostrapped set 
        (3) a boolean vector indicating which of the rows of the input data are in the out-of-bag set
        (4) a vector of the indices of the observations in the bootstrapped data set
"""
def sample_bootstraps(data_input, bootstrap_proportion=1, do_boot = True, seed = None):
           
    # length of training set
    if seed is None:
        seed = np.random.randint(1000000)


    data_input = data_input.reset_index()
    m = len(data_input)
    
    n = int((bootstrap_proportion)*float(m))
    if do_boot:
        train_i  = random.Random(seed).choices(range(m), k = n) # sampling with replacement
    else:
        train_i = range(m)
    
    is_test = ~np.in1d(range(m), train_i, assume_unique=True) 
    df_train = data_input.iloc[train_i,:]
    df_test  = data_input[is_test]    
    return df_train, df_test, is_test, train_i