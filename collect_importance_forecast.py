"""
This script collects the variable importance results of the forecasting experiments in a csv file.
"""

from helpers.import_packages import *    
from helpers.utils import *
from helpers.utils_predict import *
from helpers.utils_importance import *

experiments = []
output = []
name_ID = "test_case" 

results_raw_folder = "results/" + name_ID  + "/"
files = os.listdir(results_raw_folder)
files = [f for f in files if ".pickle" in f]


hashes = [re.findall("\d+", f)[::-1][0] for f in files] # remove non-numeric characters
hashes = list(set(hashes))
for h in hashes:
        res_perf = pk.load(open(results_raw_folder + "results_forecast" + h + ".pickle",'rb'))        
        res_perf.keys()                
        exp = res_perf["exp"]
        del exp["target"]
        exp_info = pd.DataFrame([exp])
        for shap_type in ["priority"]:
                exp_info["type_shapley"] = shap_type
                exp_info["type_experiment"] = "forecast"
                if os.path.exists(results_raw_folder + "importance_forecast_shapley_" + shap_type + "_" +  h + ".pickle"): 
                        res_shap = pk.load(open(results_raw_folder + "importance_forecast_shapley_" + shap_type + "_" +  h + ".pickle",'rb'))
                        shap_mean = res_shap["shapley_values_mean"] # shapley values
                        shap_mean.columns = ["shap_" + f for f in list(shap_mean.columns)]
                        out_add = pd.concat([res_shap["test_data"], shap_mean], axis = 1) # observed values (model input)                       
                        out_add["pred"] = aggregate_predictions(res_perf)["mean_forecast"] # forecasts
                        out_add["period"] = "forecast"
                        for k in exp_info.columns:
                                out_add[k] = exp_info[k].values[0]
                        output.append(out_add)
        
pd.concat(output).to_csv("results/aggregated/shapley_forecast_" + name_ID + ".csv")



# permutation
files = os.listdir(results_raw_folder)
hashes = [re.findall("\d+", f)[::-1][0] for f in files if "permuta" in f]
hashes = list(set(hashes))
output = []
for h in hashes:
        res_perf = pk.load(open(results_raw_folder + "results_forecast" + h + ".pickle",'rb'))        
        exp = res_perf["exp"]
        del exp["target"]
        exp_info = pd.DataFrame([exp])
        exp_info["type_experiment"] = "forecast"

        res_perm = pk.load(open(results_raw_folder + "importance_forecast_permutation_" + h + ".pickle",'rb'))
        res_perm.keys()
        out_add = pd.concat([
                res_perm["permutation_error_abs"].T, # influence of permutations on absolute error
                res_perm["permutation_error"].T, # influence of permutations on squared error
                res_perm["permutation_abs_deviance"].T # influence of permutations on change in prediction
                ])
        out_add["type"] = ["absolute_error", "squared_error", "prediction deviance"]
        for k in exp_info.columns:
                out_add[k] = exp_info[k].values[0]
        output.append(out_add)
pd.concat(output).to_csv("results/aggregated/permutation_forecast_" + name_ID + ".csv")


