
"""
This script contains helper functions to compute the variable importance
"""

from __main__ import *
import platform

"""
This function estimates Shapley values based on the SHAP package
:param model: sklearn prediction model
:param exp: dict containing experimental parameters
:param trainx: pd.DataFrame containing the features of the training set
:param testx: pd.DataFrame containing the features of the test set
:param boot_train_ix: index of the training observations that appeared in a bootstrapped sample. 
    If none the complete training set is considered
:param explainer: str indicating which method should be used to compute Shapley values: 
    The default value is priority, which means that the TreeExplainer is used for tree models, the linear explainer for linear models
    and the kernel explainer for the other models (e.g. neural network). This can be set to kernel to use the kernel
    explainer for all model families.
:return: Shapley values of the test set
"""

def compute_shapley(model, exp, train_x, test_x, boot_train_ix = None, explainer = "priority"):

    if boot_train_ix is None:
        boot_train_ix = np.arange(len(train_x))

    train_x_sub = train_x.iloc[boot_train_ix,:]
    if exp["norm_var"]:
        train_x_sub, test_x = normalise(train_x_sub, test_x)
    not_kernel_not_pca  = (not explainer == "kernel") and (exp["features"] != "pca")# we only apply the specialised 
        # explainers (Tree and linear) when "kernel" was not selected
        # as the explainer and the features have not gone though a PCA

    # TreeExplainer:
    if  not_kernel_not_pca and (exp["method"] in [
                         "Forest", "Forest_fast", 
                         "LightGBM", "LightGBM_goss",
                         "Extree", "Extree_default"]):
            shap_tree = shap.TreeExplainer(model, feature_perturbation = "interventional", data = train_x_sub)
            return(shap_tree.shap_values(test_x))
    # LinearExplainer
    elif not_kernel_not_pca and (exp["method"] in ['OLS', 'Enet', "Ridge", "Lasso"]):
        shap_linear = shap.LinearExplainer(model,
                                            masker = train_x_sub
                                            )
        out = shap_linear.shap_values(test_x)
        return(out)
    # KernelExplainer
    else:

        if exp["shapley_n_background"] >= train_x_sub.shape[0]:
            background_sample = train_x_sub
        else:
            background_sample = shap.kmeans(train_x_sub, np.min([len(train_x_sub), exp["shapley_n_background"]]))
        shap_kern = shap.KernelExplainer(model.predict,
                                        link='identity',
                                        data = background_sample)
        out = shap_kern.shap_values(test_x, l1_reg = 0, nsamples = exp["shapley_n_coalition"])
        return(out)


""" This function calculates the Shapley share coefficients for Shapley regression and does the 
    statistical inference on the Shapley values
:param data: pd.DataFrame Original input data 
:param decomp: pd.DataFrame Shapley values
:param target: str name of the response variable
:param features: names of features
:param se_type: str type of regression standard error. Needs to be compatible with statsmodels OLS
:param bootstrapping: boolean whether result os based on a bootstrapped sample, if so p-values are adjusted
:return: pd.DataFrame Shapley regression table   
"""
def shapley_regression(data, decomp, target, features,
                       se_type='HC3', bootstrapping = False):


    features_use = features

    exog_ols = data[features]
    exog_ols = sm.add_constant(exog_ols)

    exog_shapley = decomp[features_use]
    exog_shapley = sm.add_constant(exog_shapley)

    rgr_ols = OLS(endog = data[target].values, exog=exog_ols).fit(cov_type='HC3')
    rgr_shap = OLS(endog = decomp[target].values, exog=exog_shapley).fit(cov_type=se_type)
    rgr_shap.pvalues = rgr_shap.pvalues/2 # for one-sided test

    if bootstrapping:
        rgr_shap.pvalues = rgr_shap.pvalues * 2 # for one-sided test
        # we multiply p-values by 2 following Chernozhukov et al. (2017) (page 20)

    # significance level
    alpha_lvl = p_stars(rgr_shap.pvalues[features],rgr_shap.tvalues[features], no_neg = True)
    # extract results
    df_rgr = pd.DataFrame({'shap_sign' : np.sign(rgr_ols.params[features]), 
                           'coefficient' : rgr_shap.params[features],
                           'p_zero'    : np.where(rgr_shap.params[features]>0, rgr_shap.pvalues[features], 1 - rgr_shap.pvalues[features]),
                           # p_zero'    : rgr_shap.pvalues[features], 
                           'is_valid'    : rgr_shap.params[features]>0,
                           't_zero'    : rgr_shap.tvalues[features],
                           "coefficient_se": rgr_shap.bse[features], 
                           'p_lvl'       : alpha_lvl})
   

    # Shapley value share
    df_rgr['shap_share'] = shapley_share(decomp[features])


    all_cols = ['shap_sign','shap_share', 'coefficient', "coefficient_se",
                'is_valid','p_zero','p_lvl','t_zero']
    df_rgr = df_rgr[all_cols]
    return df_rgr


"""
Function to show asterisks to indicate statistical significance of regression coefficients
:param p_vals: array of p-values
:param t_vals: array of t-values
:params no_neg: boolean. If true asterisks are only shown for positive coefficients.
    This is sensible for Shapely regressions, where only positive coefficients have 
    a meaningful alignment with the response variables
"""
def p_stars(p_vals,t_vals, no_neg=False):
    # add stars
    star_list = []
    for _, (p,t) in enumerate(zip(p_vals,t_vals)):
        if (t>0) or (no_neg==False):
            if p<=0.01:
                star_list.append('***')
            elif p<=0.05:
                star_list.append('**')
            elif p<=0.1:
                star_list.append('*')
            else:
                star_list.append('')
        else:
            star_list.append('')
    return star_list

"""
This function computes the Shapley shares
:param data: pd.DataFrame for which to calculate Shapley share
:return: average Shapley share across all observations
"""
def shapley_share(data):
    data = pd.DataFrame(data)
    row_sum = np.tile(np.abs(data).sum(1).values,(data.shape[1],1)).T
    column_shares = np.abs(data)/row_sum
    column_shares = column_shares.mean(0)
    return column_shares


"""
This function computes importance measures (Shapley values and permutation importance) for the forecasting predictions
:param input_all: nested dictionary containing all the results of the forecasting experiment 
:param models: list of machine learning models
:param boot_use: int specifying how many bootstrap iterations (if model has been bootstrapped) will be used to estimate 
    Shapley values. 
:param forecast_lag: int The length of the period the model has to wait until it makes predictions. This makes sense from a policy perspective.
                     We want to make predictions not one time point ahead (a month) but longer
:param method: str specifying which importance measure is computed
"""
def compute_importance_summary(input_all, models, boot_use = None, forecast_lag = 12, method = "shapley_priority"):
    
    try:
        method_prefix, method_suffix = method.split("_")
    except:
        method_prefix = method
        method_suffix = ""

    input_res = input_all["results"]
    exp = input_all["exp"]
    features = input_all["features"]

    boot_max = input_res[list(input_res.keys())[0]]["train_fit"].shape[0]
    if boot_use is None:
        boot_use = boot_max

    if boot_use > boot_max:
        boot_use = boot_max
        warnings.warn("The maximum number of iterations that can be evaluated is " + str(boot_max) +"!", UserWarning)
   
    boot_ix = list(range(boot_use))

    shapley_all = np.empty(shape=[0, len(features)])
    shapley_mean =  pd.DataFrame()
    perm_errors_abs_mean = None
    perm_errors_squared_mean = None
    perm_abs_deviance= None
    permutation_out = []
    predictions_out = []  
    test_data_all = pd.DataFrame()

    for i in input_res.keys():

        print('|', end = '')
        horizon = exp["step_size"]
        
        # we have to skip the first observations if we have a yearly lag otherwise we would not have a true yearly forecast. 
        skip_first_obs = forecast_lag - 1

        forecast_period = np.arange(horizon) + skip_first_obs
        forecast_period = forecast_period[forecast_period < len(input_res[i]["test_pred"][0])]
        if len(forecast_period) == 0:
            continue
        
        test_set = input_res[i]["test_set"].iloc[forecast_period,:]
        test_x = test_set[features]
        test_y = test_set["target"].values        
        train_x = input_res[i]["training_set"][features]
 
        test_data_all = test_data_all.append(test_set)
        
        if method_prefix == "shapley":
        # Shapley values
            shap_values = [compute_shapley(
                model = models[i][ix],
                exp = exp,
                train_x = train_x,
                test_x = test_x,
                boot_train_ix = input_res[i]["boot_train_ix"][ix],
                explainer = method_suffix) 
                for ix in boot_ix]


            shap_values = np.hstack([shap_values])
            shap_values_flat = shap_values.reshape(-2, shap_values.shape[-1])
            shapley_all = np.vstack((shapley_all,shap_values_flat))
            shap_add = pd.DataFrame(shap_values.mean(axis = 0), index = test_set.index, columns = test_x.columns)
            shapley_mean = shapley_mean.append(shap_add) 
        # Permutation importance
        if method_prefix == "permutation":
            pp = [compute_permutation(
                model = models[i][ix],
                train_x = train_x,
                test_x = test_x,
                exp = exp,
                boot_train_ix = input_res[i]["boot_train_ix"][ix]) 
                for ix in boot_ix]
            
            permutation_out.append(np.vstack([pp]))        
            pp = [input_res[i]["test_pred"][ix, forecast_period] for ix in boot_ix]
            predictions_out.append(np.vstack([pp]))            
            
        
    if method_prefix == "permutation":
        
        permutation_out = np.hstack(permutation_out)
        predictions_out = np.hstack(predictions_out)

        true_y = test_data_all["target"].values
        perm_errors_squared = np.zeros([boot_use, len(features)])
        perm_errors_abs = np.zeros([boot_use, len(features)])
        perm_abs_deviance = perm_errors_squared.copy()
        
        for l in range(boot_use):
            err_denom_squared = compute_errors(predictions_out[l,:], true_y)["rmse_error"]
            err_denom_abs = compute_errors(predictions_out[l,:], true_y)["abs_error"]
            for f in range(len(features)):
                
                error_num_squared = [compute_errors(permutation_out[l,:,r,f], true_y)["rmse_error"] for r in range(exp["permutation_rep"])]
                error_num_squared = np.array(error_num_squared)
                
                perm_errors_squared[l,f] = np.mean(error_num_squared/err_denom_squared)
                
                error_num_abs = [compute_errors(permutation_out[l,:,r,f], true_y)["abs_error"] for r in range(exp["permutation_rep"])]
                error_num_abs = np.array(error_num_abs)
                
                perm_errors_abs[l,f] = np.mean(error_num_abs/err_denom_abs)
               
                perm_abs = [compute_errors(permutation_out[l,:,r,f], predictions_out[l,:])["abs_error"] for r in range(exp["permutation_rep"])]
                perm_abs_deviance[l,f] = np.mean(np.array(perm_abs))
        
        perm_errors_squared_mean = pd.DataFrame(perm_errors_squared.mean(0), index = test_x.columns)
        perm_errors_abs_mean = pd.DataFrame(perm_errors_abs.mean(0), index = test_x.columns)
        perm_abs_deviance = pd.DataFrame(perm_abs_deviance.mean(0), index = test_x.columns)
        
    output = {
              'exp': exp,
              'test_data': test_data_all,
              "shapley_values_mean": shapley_mean,
              "shapley_values_all": shapley_all,
              "permutation_error": perm_errors_squared_mean,
              "permutation_error_abs": perm_errors_abs_mean,
              "permutation_abs_deviance": perm_abs_deviance,
              "features": features
              }
     
    return (output)



"""
This function computes importance measures (Shapley values and permutation importance) for the out-of-bag predictions
:param input_all: nested dictionary containing all the results of the forecasting experiment 
:param models: list of machine learning models
:param boot_use: int specifying how many bootstrap iterations (if model has been bootstrapped) will be used to estimate 
    Shapley values. 
:param ix_use: string specifying the model (at a certain time t) for which we estimate Shapley values. Note that we only can investigate 
                a model at time t for which we applied bootstrapping (see parameter boot_dates in function run_experiments).
                By default this parameter is none and the model trained at the last point in time is used to estimate the Shapley values
                for all previous data points.
:param method: str specifying which importance measure is computed
"""
def compute_importance_summary_bootstrap(input_all, models, boot_use = None, ix_use = None, method = "shapley_priority"):
    try:
        method_prefix, method_suffix = method.split("_")
    except:
        method_prefix = method
        method_suffix = ""
    input_res = input_all["results"]
    exp = input_all["exp"]
    features = input_all["features"]
    n_features = len(features)
    boot_max = input_res[list(input_res.keys())[0]]["train_fit"].shape[0]


    if boot_use is None:
        boot_use = boot_max
    if boot_use > boot_max:
        boot_use = boot_max
        warnings.warn("The maximum number of iterations that can be evaluated is " + str(boot_max) +"!", UserWarning)

    if boot_use == 1:
        warnings.warn("This model was not bootstrapped. It therefore does not produce bootstrapped errors!", UserWarning)
        return None

    if ix_use is None:
        ix_use = np.array(list(input_res.keys())) # last point in time is used for bootstrapping
        ix_use = str(ix_use.astype(int).max())

    models_use = models[ix_use]
    results_use = input_res[ix_use]

    shapley_mean =  pd.DataFrame()
    train_n = len(results_use["training_set"])
    test_data_all = results_use["training_set"]

    shapley_values = np.zeros([train_n, boot_use, n_features]) * np.nan
    perm_error_out = []
    perm_abs_deviance_out = []
    permutation_abs_deviance = []
    permutation_error = []
    for i, ix in enumerate(range(boot_use)):

        training_boot_ix = results_use["boot_train_ix"][ix]
        test_boot_ix = list(set(list(range(train_n))) - set(training_boot_ix))
        test_boot_ix.sort()
        if len(test_boot_ix) == 0:
            continue
        test_x = results_use["training_set"][features].iloc[test_boot_ix]
        test_y = results_use["training_set"]["target"].values[test_boot_ix]
        train_x = results_use["training_set"][features].iloc[training_boot_ix]

        if method_prefix == "shapley":
            shapley_values[test_boot_ix,i,] = compute_shapley(model = models_use[ix],
                                                            exp = exp,
                                                            train_x = train_x,
                                                            test_x = test_x,
                                                            explainer = method_suffix)
        if method_prefix == "permutation":
            permutation_out = compute_permutation(model = models_use[ix],
                                                train_x = train_x,
                                                test_x = test_x,
                                                exp = exp)
            pred_boot = results_use["train_boot_pred"][ix, test_boot_ix]
            error_obs = compute_errors(test_y, pred_boot)["rmse_error"]

            errors_perm = []
            deviance_perm = []
            for f in range(n_features):
                errs = [compute_errors(test_y,  permutation_out[:,r, f])["rmse_error"]
                            for r in range(exp["permutation_rep"])]
                errs = np.array(errs)
                errors_perm.append((errs/error_obs).mean())
                deviance = [compute_errors(pred_boot, permutation_out[:,r, f])["abs_error"] 
                                for r in range(exp["permutation_rep"])]
                deviance = np.array(deviance)
                deviance_perm.append(deviance.mean())

            perm_error_out.append(errors_perm)
            perm_abs_deviance_out.append(deviance_perm)

    if method_prefix == "shapley": # collect shapley importance across bootstrap iterations
        shapley_mean = pd.DataFrame(np.nanmean(shapley_values, axis = 1),
                                    index = test_data_all.index,
                                    columns = test_x.columns)
    if method_prefix == "permutation": # collect permutation importance across bootstrap iterations
        permutation_error = pd.DataFrame(np.vstack(perm_error_out).mean(0), index = test_x.columns)
        permutation_abs_deviance = pd.DataFrame(np.vstack(perm_abs_deviance_out).mean(0), index = test_x.columns)

    output = {'exp': exp,
              'test_data': test_data_all,
              "shapley_values_mean": shapley_mean,
              "shapley_values_raw": shapley_values,
              "permutation_error": permutation_error,
              "permutation_abs_deviance": permutation_abs_deviance,
              "features": features
              }
    return output

"""
This function permutes the test data the output feeds into the computation of permutation importance. There exist packages that do it in a similar way.
But they do not differentiate between training and test sets. This function permutes the feature value 
of a test object using the range of values in the training set.
:param model: sklearn prediction model
:param train_x: pd.DataFrame training data
:param test_x: pd.DataFrame features of the test set
:param test_y: pd.Series response variable of the test set 
:param exp: dictionary containing experimental configuration
:param boot_train_ix: index of the length of the size of the bootstrapped sample, indicating which observations have been used when training
    on a bootstrapped samples. If None, the each training instance will be used once

"""

def compute_permutation(model, train_x, test_x, exp, boot_train_ix = None):

    if boot_train_ix is None:
        boot_train_ix = np.arange(len(train_x))

    train_x_sub = train_x.iloc[boot_train_ix,:]
    if exp["norm_var"]:
        train_x_sub, test_x = normalise(train_x_sub, test_x)
    out = []
    for i in range(len(test_x)):
        out_single = compute_permutation_single(model, 
                                                train_x_sub,
                                                np.array(test_x)[i,:].reshape(1,-1), 
                                                n_rep = exp["permutation_rep"])
        out.append(out_single)
    return(np.hstack([out]))

"""
This function permutes a single instance of the test data. It iterates through all features.
:param model: sklearn prediction model
:param train_x: pd.DataFrame training data
:param test_x_single: np.array single test instance
:param n_rep: number of permutation iterations
:returns: 
"""
def compute_permutation_single(model, train_x, test_x_single,  n_rep = 100):

    train_x = np.array(train_x)
    n_features = train_x.shape[1]

    perm_prediction = np.zeros([n_rep, n_features]) * np.nan
    for f in range(n_features):

        shuffled_set = np.tile(test_x_single, (n_rep,1))
        shuffled_set[:,f] = np.random.choice(train_x[:,f], size = n_rep, replace = True)
        perm_prediction[:,f] = model.predict(shuffled_set)



    return perm_prediction