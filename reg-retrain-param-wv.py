# Databricks notebook source
import pandas as pd
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cross_decomposition import PLSRegression as sklearnPLS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.compose import ColumnTransformer
import random
from scipy.stats import linregress
import re
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statistics import *
# import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import json
import mlflow
from databricks import automl
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
import yaml



# COMMAND ----------

workspace_url=spark.conf.get("spark.databricks.workspaceUrl")

if workspace_url=='adb-7196391053651310.10.azuredatabricks.net':
    environment='dev'
elif workspace_url=='adb-108695548543774.14.azuredatabricks.net':
    environment='tst'
elif workspace_url=='adb-7605187142077910.10.azuredatabricks.net':
    environment='prd'

# COMMAND ----------

# Load YAML file
with open('../../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

catalog=config[environment]['catalog']
schema=config[environment]['model_schema']
dataset_path=config[environment]['dataset_path']
sql_table_name=config[environment]['sql_table_name']
control_table=config[environment]['control_table_reg']
assign_alias=config[environment]['assign_alias']
artifact_path=config[environment]['model_artifact_path']

# COMMAND ----------

dbutils.widgets.text('Tx', '31AFC1_PV_CV')
Tx = dbutils.widgets.get("Tx")
# f=open("../../widgets.params.json")
# params=json.load(f)
# params=params['reg'][Tx]
# print(params)
params = spark.sql(f"SELECT * FROM {control_table} WHERE transmitter_id='{Tx}'").collect()[0].asDict()
params

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC
# MAGIC ## load widgets 

# COMMAND ----------

def get_date_range(date_range, default_start="2050-09-09", default_end="1999-09-09"):
    if date_range in ["", "NA"]:
        return default_start, default_end
    else:
        start, end = date_range.split(" to ")
        return start, end

# COMMAND ----------

# DBTITLE 1,Fetching Configs
d = lambda w: True if w=="YES" else False
level_modeling = d(params["Are_we_modeling_level?"])
area=params["area"]

automl_timeout = params["Automl_timeout"]
print(automl_timeout)
#automl_timeout=5


# COMMAND ----------

# DBTITLE 1,Load Data
# data_retrain=pd.read_pickle(dataset_path+ '/'+area+'/'+ Tx.split('_', 1)[0].replace('.', '').upper()+ '_processed.pkl').reset_index(drop=True)
data_retrain=pd.read_pickle(dataset_path+ '/'+'outputs'+'/'+ Tx.split('_', 1)[0].replace('.', '').upper()+ '_processed.pkl').reset_index(drop=True)

# COMMAND ----------

exp_name='/horizon_fd/Experiments/hor_fd_training_reg'
if mlflow.get_experiment_by_name(exp_name) is None:
    mlflow.create_experiment(name=exp_name,artifact_location=artifact_path)

mlflow.set_experiment(exp_name)
experiment_id = mlflow.get_experiment_by_name(exp_name).experiment_id

# COMMAND ----------

# DBTITLE 1,Train Model
# run automl and get the best model, that'll be the baseline model that the retrain will be performed on 
dataset = data_retrain.copy(deep=True)
with mlflow.start_run(experiment_id=experiment_id,run_name = f"Training/Re-training {Tx}", nested=True):

  SEED=42
  np.random.seed(SEED)
  random.seed(SEED)
  
  Hxs=[x for x in dataset.columns if x not in ['time', Tx] and 'lag' not in x]
  if len(dataset) > 999000:
    train_datset = dataset.sample(999000)
    mlflow.log_param("Data more than 1M", 'YES')
  else:
    train_datset = dataset
    mlflow.log_param("Data more than 1M", 'NO')
#   selected_variables = [x for x in selected_variables if x != '33FC555_PV_CV'] # to handel the exception above for 33LC554!
  if Tx != '33FC555_PV_CV':
    cols = ['time', Tx] + Hxs
    selected_variables = cols + [x for x in train_datset.columns if x[0:3] == "lag"]
    selected_variables = [x for x in selected_variables if x != '33FC555_PV_CV'] # to handel the exception above for 33LC554!
  else:
    cols = ['time', Tx] + Hxs #+  ['33FC555_PV_CV']
    selected_variables = cols + [x for x in train_datset.columns if x[0:3] == "lag"]
    selected_variables = selected_variables # to handel the exception above for 33LC554!
    

  train_datset = train_datset[[x for x in selected_variables]]
  train_datset.head().to_html(dataset_path + '/'+ area+ '/'+ Tx.split('_', 1)[0].replace('.', '').upper()+ '_retrain.html')
  # mlflow.log_artifact(dataset_path + '/'+ area+ '/'+ Tx.split('_', 1)[0].replace('.', '').upper()+ '_retrain.html')

#   # tune the automl timeout 
#   maxm = int(dataset.shape[0])
#   cutoff = int(round(0.2*dataset.shape[0], 0))
#   df_test = dataset.copy(deep=True)
#   df_test = df_test[maxm- cutoff:-1]
# train_pdf = dataset_train[0:maxm- cutoff]
  
  
  from sklearn.model_selection import train_test_split

  train_pdf, test_pdf = train_test_split(train_datset, test_size=0.03, random_state=42)
#   display(train_pdf)
  train_pdf = train_pdf.drop('time',axis=1)

mlflow.end_run()

 
summary = automl.regress(train_pdf, target_col=Tx, timeout_minutes=automl_timeout, primary_metric='mse')
 


# COMMAND ----------

model_uri=summary.best_trial.model_path
model=mlflow.pyfunc.load_model(model_uri)
model.metadata.signature

# COMMAND ----------

# DBTITLE 1,Log Best Model
with mlflow.start_run(experiment_id=experiment_id,run_name = f"best model {Tx}", nested=True):
  
  model=mlflow.pyfunc.load_model(model_uri)
  y_test = test_pdf[Tx]
  X_test = test_pdf.drop([Tx, 'time'], axis=1)

  predictions_challenger = model.predict(X_test)
  mse_challenger = mean_squared_error(y_test, predictions_challenger)
  
  mlflow.log_metric("mse", mse_challenger)
  
  signature=infer_signature(test_pdf.drop([Tx,'time'], axis=1),test_pdf[Tx])

  mlflow.sklearn.log_model(model, "best_model", signature=signature)

# COMMAND ----------

# DBTITLE 1,Registering Model
mlflow.set_registry_uri("databricks-uc")
model_name=f"{catalog}.{schema}."+"hor-fd-reg"+"-"+Tx.split('_', 1)[0].replace('.', '').upper()
mv=mlflow.register_model(model_uri,model_name)

# COMMAND ----------

# DBTITLE 1,Load Current Champion Model
try:
    mlflow.set_registry_uri("databricks-uc")
    model_name=f"{catalog}.fd_models."+"hor-fd-reg"+"-"+Tx.split('_', 1)[0].replace('.', '').upper()
    champion_model=mlflow.pyfunc.load_model(f"models:/{model_name}@Champion")
except Exception as e:
    champion_model=None
    print("No champion model exists")

# COMMAND ----------

# DBTITLE 1,Calculate Metrics
from sklearn.metrics import mean_absolute_error
if champion_model!=None:
    y_test = test_pdf[Tx]
    X_test = test_pdf.drop([Tx, 'time'], axis=1)

    predictions_challenger = model.predict(X_test)
    predictions_champion = champion_model.predict(X_test)

    mae_challenger = mean_absolute_error(y_test, predictions_challenger)
    mae_champion = mean_absolute_error(y_test, predictions_champion)

    print(mae_challenger, mae_champion)
else:
    mae_champion=float("inf")
    mae_challenger=0

# COMMAND ----------

# DBTITLE 1,Assigning Challenger/Champion Model Alias
if assign_alias==True:
    if mae_challenger < mae_champion:
        mlflow.set_registry_uri("databricks-uc")
        client=MlflowClient()
        client.set_registered_model_alias(model_name, "Champion",int(mv.version))
        print("Champion model assigned")
