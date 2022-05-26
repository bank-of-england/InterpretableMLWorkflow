from helpers.import_packages import *
import helpers.config as config  
from helpers.utils import *
from helpers.utils_predict import *
from helpers.utils_importance import *

experiments = []
name_ID = "test_case"
output = []

results_raw_folder = "results/" + name_ID  + "/"
files = os.listdir(results_raw_folder)
files = [f for f in files if "out-of-bag" in f]
hashes = [re.findall("\d+", f)[::-1][0] for f in files] # remove non-numeric characters
hashes = list(set(hashes))

for h in hashes:
        res_perf = pk.load(open(results_raw_folder + "results_out-of-bag" + h + ".pickle",'rb'))        
        exp = res_perf["exp"]
        del exp["target"]        
        exp_info = pd.DataFrame([exp]).copy()
        
        for shap_type in ["priority"]:
                exp_info["type_shapley"] = shap_type
                exp_info["type_experiment"] = type
                for period in ["1999-12-01", "2008-08-01", "2019-11-01"]:
                        if os.path.exists(results_raw_folder + "importance_oob_shapley_" + shap_type + "_period_" + period + "_" + h + ".pickle"):
                                res_shap = pk.load(open(results_raw_folder + "importance_oob_shapley_" + shap_type + "_period_" + period + "_" + h + ".pickle",'rb'))
                                
                                shap_mean = res_shap["shapley_values_mean"]
                                shap_mean.columns = ["shap_" + f for f in list(shap_mean.columns)]
                                out_add = pd.concat([res_shap["test_data"], shap_mean], axis = 1)
                                # cut sample
                                ix_max = int(np.where(out_add.index == period)[0])
                                out_add = out_add.iloc[0:(ix_max + 1),:]

                                pred = res_perf["results"][list(res_perf["results"].keys())[::-1][0]]
                                pred = np.nanmean(pred["train_boot_pred"], 0)
                                pred = pred[0:(ix_max + 1)]
                                out_add["pred"] = pred
                                out_add["period"] = period
                                for k in exp_info.columns:
                                        out_add[k] = exp_info[k].values[0]
                                output.append(out_add)

pd.concat(output).to_csv("results/aggregated/shapley_out_of_bag_" + name_ID + ".csv")
