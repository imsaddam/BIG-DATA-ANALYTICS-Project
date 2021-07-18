!pip install pyspark

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import sklearn
import random
import os
from pyspark.sql.functions import *
from pyspark import SparkContext
from pyspark.sql import SparkSession 
from pyspark.ml  import Pipeline     
from pyspark.sql import SQLContext  
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer



spark = SparkSession \
.builder \
.appName("Covid 19 Data Analysis with pyspark") \
.config("spark.some.config.option", "some-value") \
.getOrCreate()

#Create a main data frame from using data source

df = spark.read.format('com.databricks.spark.csv').\
options(header='true', \
inferschema='true').\
load("/FileStore/tables/*.csv",header=True)

df.printSchema()

#create a view of data Frame

df.createOrReplaceTempView("covid")

display(df)


#seperate data Frame created for cofirmed, deaths, active etc.

confirmed_dataFrame = spark.read.format('com.databricks.spark.csv').\
options(header='true', \
inferschema='true').\
load("/FileStore/tables/confirmed_data/time_series_covid19_confirmed_global.csv",header=True)

deaths_dataFrame = spark.read.format('com.databricks.spark.csv').\
options(header='true', \
inferschema='true').\
load("/FileStore/tables/deaths_data/time_series_covid19_deaths_global.csv",header=True)


latest_data = spark.read.format('com.databricks.spark.csv').\
options(header='true', \
inferschema='true').\
load("/FileStore/tables/latest_data/08_19_2020.csv",header=True)

recoveries_dataFrame = spark.read.format('com.databricks.spark.csv').\
options(header='true', \
inferschema='true').\
load("/FileStore/tables/recoveries_data/time_series_covid19_recovered_global.csv",header=True)

apple_mobility = spark.read.format('com.databricks.spark.csv').\
options(header='true', \
inferschema='true').\
load("/FileStore/tables/apple_mobility/applemobilitytrends_2020_08_18.csv",header=True)



# Convert all the data frame into pandas 
confirmed_data = confirmed_dataFrame.toPandas()
deaths_data = deaths_dataFrame.toPandas()
latest_cases = latest_data.toPandas()
recovered_data = recoveries_dataFrame.toPandas()
apple_data = apple_mobility.toPandas()

confirmed_data

deaths_data
latest_cases
recovered_data
apple_data

columns = confirmed_data.keys()

confirmed = confirmed_data.loc[:, columns[4]:columns[-1]]
deaths = deaths_data.loc[:, columns[4]:columns[-1]]
recoveries = recovered_data.loc[:, columns[4]:columns[-1]]


dates = confirmed.keys()
total_world_cases = []
total_world_deaths = [] 
total_mortality_rate = []
total_recovery_rate = [] 
total_recovered = [] 
world_total_active = [] 

for i in dates:
    sum_of_confirmed_cases = confirmed[i].sum()
    sum_of_total_deaths = deaths[i].sum()
    sum_of_recovered = recoveries[i].sum()
    
    # confirmed, deaths, recovered, and active
    total_world_cases.append(sum_of_confirmed_cases)
    total_world_deaths.append(sum_of_total_deaths)
    total_recovered.append(sum_of_recovered)
    world_total_active.append(sum_of_confirmed_cases-sum_of_total_deaths-sum_of_recovered)
    
    # calculate rates
    total_mortality_rate.append(sum_of_total_deaths/sum_of_confirmed_cases)
    total_recovery_rate.append(sum_of_recovered/sum_of_confirmed_cases)



    def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

def average_changes(data, window_size):
    average_changes = []
    for i in range(len(data)):
        if i + window_size < len(data):
            average_changes.append(np.mean(data[i:i+window_size]))
        else:
            average_changes.append(np.mean(data[i:len(data)]))
    return average_changes

# window size
window = 7

# confirmed cases
world_daily_increase = daily_increase(total_world_cases)
world_confirmed_avg= average_changes(total_world_cases, window)
world_daily_increase_avg = average_changes(world_daily_increase, window)

# deaths
world_daily_death = daily_increase(total_world_deaths)
world_death_avg = average_changes(total_world_deaths, window)
world_daily_death_avg = average_changes(world_daily_death, window)


# recoveries
world_daily_recovery = daily_increase(total_recovered)
world_recovery_avg = average_changes(total_recovered, window)
world_daily_recovery_avg = average_changes(world_daily_recovery, window)


# active 
world_active_avg = average_changes(world_total_active, window)


days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
total_world_cases = np.array(total_world_cases).reshape(-1, 1)
total_world_deaths = np.array(total_world_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)


prediction_days = 10
future_predictions = np.array([i for i in range(len(dates)+prediction_days)]).reshape(-1, 1)
updated_date = future_predictions[:-10]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_predictions_dates = []
for i in range(len(future_predictions)):
    future_predictions_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


updated_date = updated_date.reshape(1, -1)[0]
plt.figure(figsize=(15, 12))
plt.plot(updated_date, total_world_cases)
plt.plot(updated_date, world_confirmed_avg, linestyle='dashed', color='orange')
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Start Date 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Worldwide Coronavirus Cases', 'Moving Average {} Days'.format(window)], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()



plt.figure(figsize=(15, 12))
plt.plot(updated_date, total_world_deaths)
plt.plot(updated_date, world_death_avg, linestyle='dashed', color='red')
plt.title('Number of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Start Date 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Worldwide Coronavirus Deaths', 'Moving Average {} Days'.format(window)], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


plt.figure(figsize=(15, 12))
plt.plot(updated_date, total_recovered)
plt.plot(updated_date, world_recovery_avg, linestyle='dashed', color='green')
plt.title('Number of Coronavirus Recoveries Over Time', size=30)
plt.xlabel('Start Date 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Worldwide Coronavirus Recoveries', 'Moving Average {} Days'.format(window)], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()



plt.figure(figsize=(15, 12))
plt.plot(updated_date, world_total_active)
plt.plot(updated_date, world_active_avg, linestyle='dashed', color='Yellow')
plt.title('Number of Coronavirus Active Cases Over Time', size=30)
plt.xlabel('Start Date 1/22/2020', size=30)
plt.ylabel('Number of Active Cases', size=30)
plt.legend(['Worldwide Coronavirus Active Cases', 'Moving Average {} Days'.format(window)], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()



plt.figure(figsize=(15, 12))
plt.bar(updated_date, world_daily_increase)
plt.plot(updated_date, world_daily_increase_avg, color='OrangeRed', linestyle='dashed')
plt.title('World Daily Increases in Confirmed Cases', size=30)
plt.xlabel('Start Date 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Moving Average {} Days'.format(window), 'World Daily Increase in COVID-19 Cases'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


plt.figure(figsize=(15, 12))
plt.bar(updated_date, world_daily_death)
plt.plot(updated_date, world_daily_death_avg, color='Red', linestyle='dashed')
plt.title('World Daily Increases in Confirmed Deaths', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Moving Average {} Days'.format(window), 'World Daily Increase in COVID-19 Deaths'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 10))
plt.bar(updated_date, world_daily_recovery)
plt.plot(updated_date, world_daily_recovery_avg, color='Green', linestyle='dashed')
plt.title('World Daily Increases in Confirmed Recoveries', size=30)
plt.xlabel('Start Date 1/22/2020', size=30)
plt.ylabel('Number of Cases', size=30)
plt.legend(['Moving Average {} Days'.format(window), 'World Daily Increase in COVID-19 Recoveries'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(15, 11))
plt.plot(total_recovered, total_world_deaths)
plt.title('Number of Coronavirus Deaths vs. # of Coronavirus Recoveries', size=30)
plt.xlabel('Number of Coronavirus Recoveries', size=30)
plt.ylabel('Number of Coronavirus Deaths', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()



print("Bangladesh Coronavirus Cases...")
query = """
SELECT
    Country_Region, Last_Update, Confirmed, Deaths, Recovered, Active, Incidence_Rate
   
FROM
    covid
    
WHERE Country_Region in ('Bangladesh') 
"""
spark.sql(query).show()



print("Luxembourg Coronavirus Cases...")
query = """
SELECT
    Country_Region, Last_Update, Confirmed, Deaths, Recovered, Active, Incidence_Rate
   
FROM
    covid
    
WHERE Country_Region in ('Luxembourg') 
"""
spark.sql(query).show()