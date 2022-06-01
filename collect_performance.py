"""
This script collects the performance results of the forecasting experiments in a single csv files.
"""
from helpers.import_packages import * 
import helpers.config as config
from helpers.utils import *
from helpers.utils_predict import *
from helpers.utils_importance import *

predictions = {}
experiments = []
name_ID = "test_case" 

results_raw_folder = "results/" + name_ID  + "/"
files = os.listdir(results_raw_folder)
files = [f for f in files if "results_" in f]
files = [f for f in files if "forecast" in f]
for ix in files:
        print("|", end = "")
        results = pk.load(open(results_raw_folder + ix,'rb'))
        results["exp"]["hash"] = "hash_" + str(results["exp"]["hash"])
        res = aggregate_predictions(results, forecast_lag = results["exp"]["lag"])
        experiments.append(results["exp"])
        predictions[results["exp"]["hash"]] = res


experiments_tab = pd.DataFrame(experiments)
experiments_tab.index = experiments_tab.hash

output_forecast = prediction_table(predictions, experiments_tab, config.periods)
output_forecast["pred_all"].to_csv("results/aggregated/pred_all_" + name_ID + "_raw.csv")





