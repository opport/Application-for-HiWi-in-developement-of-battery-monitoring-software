# -*- coding: utf-8 -*-
"""
Temperature Data Task

@author: Thanuj Singaravelan
"""


import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


output_folder = Path(os.getcwd()) / "output/"
Path(output_folder).mkdir(exist_ok=True)


"""
### to_series_interpolate() ###

Arguments:
    - data : imported .csv file as a pandas dataframe
    - value : column name to consider and convert to time series
    - interval : interval for interpolation in minutes
    - intrp_method : interpolation method from scipy
    
Outputs:
    - data: interpolated time series dataframe w.r.t value with datetime index
    - limits: limits of value (used for plotting [y axis limits])
    
Summary:
    This function takes in the raw .csv file, converts timestamp column to datetime index
    and converts the dataframe into a timeseries dataframe w.r.t. a given value
"""
def to_series_interpolate(data,value='Wert',interval=15,intrp_method='akima'):

    data['Zeitstempel']=pd.to_datetime(data['Zeitstempel'],format='%Y%m%d%H%M')
    data = pd.DataFrame(data,columns=['Zeitstempel',value])
    data.set_index('Zeitstempel',inplace=True)

    data = data.resample(str(interval)+'min').interpolate(method=intrp_method)

    ### safety margin of 1.25 to ensure all data point are visible when plotting ###
    limits = [1.25*(data[data.columns[0]].min()),1.25*(data[data.columns[0]].max())]


    return data,limits



"""
### max_min_yearly() ###

Arguments:
    - data : timeseries dataframe w.r.t temperature with datetime index

Outputs:
    - max_temp : timeseries dataframe with max temperature by each year with datetime index
    - min_temp : timeseries dataframe with min temperature by each year with datetime index
    
Summary:
    This function takes in the timeseries dataframe w.r.t a value with datetime index
    and returns the max and min temperature values by year as 2 timeseries dataframes
"""
def max_min_yearly(data):

    ### finding all years in the given data ###
    years = sorted(list(set(data.index.year)))

    max_temp = pd.DataFrame()
    min_temp = pd.DataFrame()

    for y in years:

        max_temp = max_temp.append(data.loc[[data[str(y)][data.columns[0]].idxmax()]])
        min_temp = min_temp.append(data.loc[[data[str(y)][data.columns[0]].idxmin()]])

    return max_temp,min_temp



"""
### max_min_day_plot() ###

Arguments:
    - data : timeseries dataframe w.r.t temperature with datetime index

Outputs:
    - figure saved in working directory : a plot of the temperatures for the hottest and coldest 
    days of each provided year onto the same axis, over time of day
    
Summary:
    This function takes in the timeseries dataframe w.r.t a value with datetime index
    and returns a plot of the temperatures for the hottest and coldest
    days of each provided year onto the same axis, over time of day
"""
def max_min_day_plot(data):

    ### finding all years in the given data ###
    years = sorted(list(set(data.index.year)))

    fig = plt.figure(figsize=(10,10),tight_layout=True)
    plt.suptitle('Hottest and Coldest Days vs Time of Day by Year')

    for y in years:

        max_day = data[data[str(y)][data.columns[0]].idxmax().strftime('%Y%m%d')]
        ### convert time stamp index to '%H:%M:%S' since working within a day ###
        max_day.index = max_day.index.strftime('%H:%M:%S')

        min_day = data[data[str(y)][data.columns[0]].idxmin().strftime('%Y%m%d')]
        ### convert time stamp index to '%H:%M:%S' since working within a day ###
        min_day.index = min_day.index.strftime('%H:%M:%S')

        ax = fig.add_subplot(2, 1, 1)
        ax.plot(max_day,'-',label=data[str(y)][data.columns[0]].idxmax().strftime('%Y-%m-%d'))
        ax.legend(loc="best")
        ax.set_title("Hottest Days vs Time of Day")
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.columns[0])
        ax.set_ylim(limits[0],limits[1])
        ax.xaxis.set_major_locator(plt.MaxNLocator(24))
        plt.xticks(rotation=45)

        ax = fig.add_subplot(2, 1, 2)
        ax.plot(min_day,'--',label=data[str(y)][data.columns[0]].idxmin().strftime('%Y-%m-%d'))
        ax.legend(loc="best")
        ax.set_title("Coldest Days vs Time of Day")
        ax.set_xlabel(data.index.name)
        ax.set_ylabel(data.columns[0])
        ax.set_ylim(limits[0],limits[1])
        ax.xaxis.set_major_locator(plt.MaxNLocator(24))
        plt.xticks(rotation=45)

    fig.savefig('Hottest and Coldest Days vs Time of Day by Year')
    fig.show()



"""
### to_csv_file() ###

Arguments:
    - data_list : list of dataframes to export as .csv to working directory
    - data_list_names : list of names(str) of dataframes to export as .csv to working directory

Outputs:
    - .csv files saved in working directory : saves .csv files based on given data frames
    
Summary:
    This is a generic function that takes in a list of dataframes and exports it as a .csv file to
    an output folder in the working directory
"""
def to_csv_file(data_list,data_list_names):

    for dataframe,dataframe_name in zip(data_list,data_list_names):
        dataframe.to_csv(output_folder / (dataframe_name + '.csv'))



"""
### to_plotter() ###

Arguments:
    - data_list : list of dataframes to export as figures to working directory
    - data_list_names : list of names(str) of dataframes to export as figures to working directory

Outputs:
    - figures saved in working directory : saves as figures based on given data frames
    
Summary:
    This is a generic function that takes in a list of dataframes and exports it as figures to working directory
"""
def to_plotter(data_list,data_list_names):


    for dataframe,dataframe_name in zip(data_list,data_list_names):


        fig,ax = plt.subplots(figsize=(10,5),tight_layout=True)
        plt.suptitle(dataframe_name)
        plt.plot(dataframe,'r-')
        plt.ylim(limits[0],limits[1])
        plt.xlabel(dataframe.index.name)
        plt.ylabel(dataframe.columns[0])
        date_form = DateFormatter("%b-%Y")
        ax.xaxis.set_major_formatter(date_form)
        ax.xaxis.set_major_locator(plt.MaxNLocator(24))
        plt.xticks(rotation=45)
        plt.savefig(output_folder / dataframe_name)
        plt.show()



### BONUS ###



"""
### train_sarimax() ###

Arguments:
    - data : timeseries dataframe w.r.t temperature with datetime index

Outputs:
    - model_fit : trained SARIMAX model based on given data
    - residual : pandas series of residual error for trained model
    - figure saved in working directory : saves figure of SARIMAX predictions and residuals
    
Summary:
    This function is for training a SARIMAX time series model:
    - gets timeseries dataframe w.r.t temperature with datetime index and interpolates to weekly basis
    - splits data into training and test set based on ratio - train_test_ratio = 0.85 (default)
    - trains SARMIAX model {(1,1,1),(1,1,1),52} based on training set
    [note: parameters for model were determined by ACF, PACF, seasonal decomposition tests and
     seasonality was chosen on a yearly basis - this was performed offline in another program]
    - calculates residual based on test set on the trained model
    - saves figure of SARIMAX model and residuals to working directory
"""
def train_sarimax(data):

    week_series = data['Wert'].resample('W').mean()

    train_test_ratio = 0.85

    train_data = week_series.iloc[:int(train_test_ratio*(len(week_series)))]
    test_data = week_series.iloc[int(train_test_ratio*(len(week_series))):]

    model = SARIMAX(train_data,order=(1,1,1),seasonal_order=(1,1,1,52))
    model_fit = model.fit()


    start = len(train_data)

    ### define end as global variable to be used by predict_sarimax() ###
    global end
    end = len(train_data) + len(test_data) - 1

    prediction_train = model_fit.predict(0, start).rename('Wert')
    prediction_test = model_fit.predict(start, end).rename('Wert')

    prediction_total = pd.concat([prediction_train,prediction_test])

    residual = week_series - prediction_total
    residual.index.name = 'Zeitstempel'

    fig = plt.figure(figsize=(10,10),tight_layout=True)
    plt.suptitle('SARIMAX Weekly Model')

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(week_series,'k-',label='Actual')
    ax.plot(prediction_train,'b-',label='Prediction - Train Set')
    ax.plot(prediction_test,'r-',label='Prediction - Test Set')
    ax.axvline(x=prediction_test.index[0],color='k',ls='--')
    ax.legend(loc="best")
    ax.set_title("SARIMAX Model - Training vs Test vs Actual")
    ax.set_ylim(limits[0],limits[1])
    ax.set_xlabel(data.index.name)
    ax.set_ylabel(data.columns[0])
    date_form = DateFormatter("%b-%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.xticks(rotation=45)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(residual,'k-',label='Residuals')
    ax.axvline(x=prediction_test.index[0],color='k',ls='--')
    ax.legend(loc="best")
    ax.set_title("SARIMAX Model - Residuals")
    ax.set_ylim(limits[0],limits[1])
    ax.set_xlabel(data.index.name)
    ax.set_ylabel(data.columns[0])
    date_form = DateFormatter("%b-%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.xticks(rotation=45)

    fig.savefig(output_folder / 'SARIMAX Weekly Model')
    fig.show()

    return model_fit,residual



"""
### predict_sarimax() ###

Arguments:
    - data : timeseries dataframe w.r.t temperature with datetime index (used for plotting purpose only)
    - model_fit : trained model object given by train_sarimax()
    - weeks : number of weeks to predict from the time recorded data ends
    - res : resampling frequency for predicted data, based on linear interpolation {'D'-day,'W'-week, etc.}

Outputs:
    - prediction : pandas timeseries with predicted values with frequency res
    - figure saved in working directory : saves figure of SARIMAX predictions
    
Summary:
    This function is for predciting using a trained SARIMAX model:
    - gets timeseries dataframe w.r.t temperature with datetime index and interpolates to weekly basis
    - applies trained SARMIMAX model and predicts the output as a pandas timeseries
    - resamples according to input using linear interpolation
    - saves figure of SARIMAX predictions to working directory
"""
def predict_sarimax(data,model_fit,weeks=10,res='D'):

    week_series = data['Wert'].resample('W').mean()

    prediction = model_fit.predict(end, end+weeks).rename('Wert')
    prediction = prediction.resample(res).interpolate(method='linear')
    prediction.index.name = 'Zeitstempel'

    fig = plt.figure(figsize=(10,10),tight_layout=True)
    plt.suptitle('SARIMAX Prediction')

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(week_series,'k-',label='Actual')
    ax.plot(prediction,'r-',label='Prediction weeks : '+str(weeks))
    ax.axvline(x=prediction.index[0],color='k',ls='--')
    ax.legend(loc="best")
    ax.set_title("SARIMAX Prediction for "+str(weeks)+" weeks")
    ax.set_ylim(limits[0],limits[1])
    ax.set_xlabel(data.index.name)
    ax.set_ylabel(data.columns[0])
    date_form = DateFormatter("%b-%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.xticks(rotation=45)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(prediction,'ro-',label='Prediction weeks : '+str(weeks))
    ax.legend(loc="best")
    ax.set_title("SARIMAX Prediction for "+str(weeks)+" weeks - Zoomed")
    ax.set_ylim(limits[0],limits[1])
    ax.set_xlabel(data.index.name)
    ax.set_ylabel(data.columns[0])
    date_form = DateFormatter("%d-%b-%Y")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))
    plt.xticks(rotation=45)

    fig.savefig(output_folder / 'SARIMAX Prediction')
    fig.show()

    return prediction



### MAIN FUNCTION ###
"""
note: execution of code might take around 1 to 2 mins since SARIMAX model takes some time to train
DO NOT PANIC IF CODE SEEMS STUCK
"""


if __name__ == '__main__':

    data_temp = pd.read_csv("temperatures.csv")

    ### define limits as global variable for plotting inside all functions ###
    global limits
    series_temp,limits = to_series_interpolate(data_temp)


    max_yearly,min_yearly = max_min_yearly(series_temp)

    max_min_day_plot(series_temp)

    to_csv_file([series_temp,max_yearly,min_yearly],['Interpolated Temperature','Max Yearly Temperature','Min Yearly Temperature'])
    to_plotter([series_temp,max_yearly,min_yearly],['Interpolated Temperature vs Time','Max Yearly Temperature vs Time','Min Yearly Temperature vs Time'])

    model_sarimax,residuals = train_sarimax(series_temp)

    pred_series = predict_sarimax(series_temp,model_sarimax,weeks=10,res='D')

