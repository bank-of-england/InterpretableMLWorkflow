from helpers.import_packages import *     # standard libraries
import helpers.config as config    # set target variable, model choice and sensitivity analysis options
from helpers.utils import *
from helpers.utils_importance import *
from helpers.utils_predict import *
config.n_cpus = 6 # set the number of threads for parallelisation (mostly for hpyerparameter serach)

#%% loads dictionary that specifies the set of experiments
exp_setup_filepath = os.path.join('setup_files/main.yaml')
with open(exp_setup_filepath, 'r') as outfile:
    experiment_parameters = yaml.load(outfile)

output_folder = "results/test_case/"
experiment_parameters["output_folder"] = output_folder

#%% read data
data_file = config.data_path + config.datafile +'.csv'
raw_data = pd.read_csv(data_file)
# transformations of the variables are specified in the first row of the FRED MD file.
transformations = dict(zip(raw_data.columns[1:], raw_data.iloc[0,].values[1:].astype(int))) 
raw_data = raw_data.drop([0]).reset_index(drop = True)
raw_data = raw_data.rename(columns = {"sasdate": "date"}) 
raw_data.date = pd.to_datetime(raw_data.date, format = "%m/%d/%Y")

# exclude variables that have more missings than the minimum number of missings
proprotion_missing = raw_data.isnull().sum() / len(raw_data)
variables_keep = list(proprotion_missing.index[proprotion_missing == proprotion_missing.min()])
raw_data = raw_data[variables_keep]
raw_data = raw_data.dropna().reset_index(drop = True)

transformations = {v:transformations[v] for v in variables_keep[1:]} 

# remove special characters from column names | some models cannot handle them
for key in list(transformations.keys()):
    key_replace = re.sub('[^A-Za-z0-9]+', ' ', key)
    if key != key_replace:
        raw_data.rename({key: key_replace}, axis=1, inplace=True)
        transformations[key_replace] = transformations.pop(key)

experiments = list_experiments(experiment_parameters) # this create a dataframe of all experiments that we want to run
experiments = remove_duplicates(experiments) # this removes experiments that have effectively the same specification because a parameter is varied that is not relevant for the specific set-up


# forecasting
run_experiments(raw_data, 
                        experiments, 
                        transformations, 
                        importance_tasks=["shapley_priority", "permutation"],
                        test_time = ["forecast"],
                        out_of_bag_dates = None,
                        overwrite = False)
                        
# out-of_bag
run_experiments(raw_data, 
                        experiments, 
                        transformations, 
                        importance_tasks = ["shapley_priority", "permutation"],
                        test_time = ["out-of-bag"],
                        out_of_bag_dates = ['1999-12-01', '2008-08-01', '2019-11-01'],
                        overwrite = False)





