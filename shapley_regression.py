from helpers.import_packages import * 
import helpers.config as config    
from helpers.utils import *
from helpers.utils_predict import *
from helpers.utils_importance import *

name_ID = "test_case" 
                       
dat = pd.read_csv("results/aggregated/shapley_forecast_" + name_ID + ".csv")
dat.index = pd.to_datetime(dat.date.values)
dat.index.name = 'date'
shapley_tables = {}



method = "LightGBM" 

ix = (dat["method"].values == method) & \
                (dat.index >= np.datetime64(config.periods["all"][0])) & \
                (dat.index <= np.datetime64(config.periods["all"][1]))
features_use = config.features_key + ["UNRATE"] # add lagged response feature as well
obs_values = dat.iloc[ix,:][["target"] + features_use] # observed feature values
features_shap = ["shap_" + f for f in features_use] 
shap_values = dat.iloc[ix,:][["target"] + features_shap] # shapley values
shap_values.columns = [re.sub("shap_", "", f) for f in list(shap_values.columns)]

# average values when we have estimated the models several times (with different random seeds)
obs_values = obs_values.groupby("date").mean() 
shap_values = shap_values.groupby("date").mean()

# compute shapley regression
table = shapley_regression(
                        data = obs_values, 
                        decomp = shap_values,
                        target = "target", 
                        features = features_use, 
                        se_type = 'HC3'
                        )
        