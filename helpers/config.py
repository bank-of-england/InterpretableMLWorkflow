"""
This file provides specifies sone basic configuration that is not expected to change between different experiments
"""

import os
import pandas as pd
import multiprocessing

datafile = '2021-08';         

n_cpus = min([multiprocessing.cpu_count(), 4]) # number of CPU threads used
main_path = os.getcwd() + "/"
data_path = main_path + 'data/'      

# See McCracken & Ng (2015)
features_key = [
                "RPI", 
                "INDPRO",
                "M2SL",
                "CPIAUCSL",
                "S P 500", 
                "TB3MS",
                "OILPRICEx",
                "BUSLOANS",
                "DPCERA3M086SBEA"
                ]



time_var = 'date' 
start_time = pd.to_datetime('1959-01-01', format = "%Y-%m-%d")
end_time = pd.to_datetime('2019-11-01', format = "%Y-%m-%d")
date_type = "month"

periods = {
    "all": ["1990-01-01", "2019-11-01"],
    "pre_crisis": ["1990-01-01", "2008-08-01"],
    "pre_mid_90s": ["1990-01-01", "1995-12-01"],
    "post_mid_90s": ["1996-01-01", "2019-11-01"],
    "01/1990 - 12/1999": ["1990-01-01", "1999-12-01"],
    "01/2000 - 08/2008": ["2000-01-01", "2008-08-01"],
    "09/2008 - 11/2019":["2008-09-01", "2019-11-01"]
}


