
This reposity contains the code used in the Bank of England Staff Working Paper 984 _An interpretable machine learning workflow with an application to economic forecasting_ by Marcus Buckmann and Andreas Joseph.

In the paper, we propose a generic workﬂow for the use of machine learning models to inform decision making and to communicate modelling results with stakeholders. It
involves three steps: (1) a comparative model evaluation, (2) a feature importance analysis and (3) statistical inference based on Shapley value decompositions. The paper 
demonstrates each step by forecasting changes in US unemployment one year ahead using [FRED-MD](https://research.stlouisfed.org/wp/more/2015-012), a US data base of monthly macroeconomic measurements.

The code is not intended as a stand-alone package. It can be used to reproduce the results of the paper. But as the porposed workflow is quite general, parts of it may be transfered to other applications. No warranty is given. Please consult the licence file.


Should you have any questions or spot a bug in the code, please send an email to marcus.buckmann@bankofengland.co.uk or raise an issue within the repository.


The code has been developed under Python 3.9.7 Anaconda distribution.

THe file python_env.yml specifies the Anaconda environment in which the experiments were run.

# Structure of the code





# Disclaimer
This package is an outcome ofa research project. All errors are those of the authors. All views expressed are personal views, not those of any employer.
