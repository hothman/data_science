import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
from scipy.optimize import curve_fit

data = pd.read_csv("../data/time_series_19-covid-Confirmed.csv")

def clean_data(dataframe, country):
    mydata = dataframe[dataframe['Country/Region'] == country].drop(columns=["Province/State", 
            "Country/Region", "Lat", "Long"]).reset_index(drop=True).T
    mydata.rename(columns={0:"cumulative_count"}, inplace=True)
    mydata["date"] = pd.to_datetime( mydata.index )
    mydata = mydata[mydata["cumulative_count"]!=0]
    mydata["days"] = np.arange(1, len(mydata)+1)
    mydata.reset_index(drop=True, inplace=True)   
    return mydata 

def SPG(t, r, m, A):
    """ """
    return (  ((r/m)*np.array(t)  ) +A )**m

def expgrowth(t, alpha, beta): 
    """ exponential growth"""
    return alpha*np.exp( np.array(t))**beta

def curve_fitting(dataframe): 
    days = dataframe.days
    cases_count = dataframe.cumulative_count
    parameters, pcov = curve_fit(expgrowth, days, cases_count)
    return parameters, pcov

def plot_exp_forecast(dataframne, popt, forecast_offset=5, color="red"): 
    plot1 = plt.plot(dataframne.days, dataframne.cumulative_count,'o'  ,color =color)
    offset_array = np.arange(list(dataframne.days)[-1]+1, list(dataframne.days)[-1]+forecast_offset+1  )
    offset = np.append(dataframne.days, offset_array )
    plot2 =  plt.plot(offset, expgrowth(offset, *popt),"--",color =color)  
    print(offset)
    return plot1, plot2

import random 
import matplotlib.colors as col 
import itertools
from datetime import datetime, timedelta

def random_col(alpha=1, keep_alpha=False):
    r = random.random()
    g = random.random()
    b = random.random()
    return col.to_hex((r, g, b, alpha), keep_alpha=keep_alpha)
    
def wrap(countries, dataframe, forecast=5):
    number_of_panels = len(countries)
    fig, axs = plt.subplots(figsize=(10, 5) )  
    
    for country in countries: 
        color =  random_col()
        idx = countries.index(country)
        clean = clean_data(dataframe, country)
        forecasted_date = list(clean.date)[-1]+timedelta(days=forecast)
        #timedelta(days=forecast)
        popt, pcov = curve_fitting(clean)
        alpha = round(popt[0],3) 
        beta = round(popt[1],3) 
        day_list = list(clean["days"])
        forecast_day = day_list[-1] + forecast
        cases_forecasted_day = expgrowth(t=forecast_day, alpha=alpha, beta=beta) 
        
        fig.suptitle('COVID-19 forecast for {0}'.format(country), fontsize=14, fontweight='bold')
        plt.xlabel("Days", fontsize=18)
        plt.ylabel("Number of infected people", fontsize=18)
        plot1, plot2 = plot_exp_forecast(clean, popt, forecast, color = color )
        # text to insert in the figure
        text= r"""
        Country: {6}
        $\beta = {2}$, $\alpha = {3}$.
        Date of the start of the outbreak:{0}
        Forcast until {1}
        Prediction: On date {4} we will reach {5} cases.
        """.format(clean.date[0].strftime("%Y-%m-%d"),
                   forecasted_date.strftime("%Y-%m-%d"),
                   beta, alpha, 
                   forecasted_date.strftime("%Y-%m-%d"),
                   int( round(cases_forecasted_day,0)), 
                  country)
        text2 = """Author: Houcemeddine Othman
                   An exponential model is used described in Chen and Yu (2020, DOI 10.1186/s41256-020-00137-4)
                   for more details check the python code on my jupyter notebook at:
                   https://github.com/hothman/data_science/blob/master/covid-19/notebook/Untitled.ipynb"""
                

        plt.scatter(forecast_day, cases_forecasted_day, color=color, marker = 's',s=100 )
        plt.text(0, -150, text, ha='left', wrap=True, fontsize=14, family='serif', 
                bbox=dict(boxstyle="round", fc="none") )
        plt.text(0, -190, text2 , ha='left', wrap=True, fontsize=12, family='serif')
        
        plt.grid( linestyle='-.', linewidth=0.5, which='both')
        
        start_day = list(clean.days)[0]

        cases_on_start_day = list(clean.cumulative_count)[0]
        x_annot_box1 = start_day+1
        y_annot_box1 = cases_on_start_day+30
        axs.annotate('Start of \n outbreak: {0} '.format(clean.date[0].strftime("%d-%m-%Y")) , 
            xy=(start_day, cases_on_start_day), 
            xytext=(x_annot_box1, y_annot_box1),
            fontsize=12,
            arrowprops=dict(facecolor='#B9B9B9', shrink=0.05))
        
        end_data = list(clean.days)[-1]
        end_data_cases = list(clean.cumulative_count)[-1]
        x_annot_box3 = end_data - 6
        y_annot_box3 = end_data_cases+6
        axs.annotate('Last data point: \n     {0} '.format(clean.date[len(clean.date)-1].strftime("%d-%m-%Y")) , 
        xy=(end_data-0.5, end_data_cases), 
        xytext=(x_annot_box3, y_annot_box3),
        fontsize=12,
        arrowprops=dict(facecolor='#B9B9B9', shrink=0.05))
        
        x_annot_box2 = forecast_day-5
        y_annot_box2 = cases_forecasted_day-25
        axs.annotate('{0} cases are \n predicted \n in {1}'.format(int( round(cases_forecasted_day,0)),
            forecasted_date.strftime("%d-%m-%Y")) , 
            xy=(forecast_day-0.2, cases_forecasted_day),
            xytext=(x_annot_box2, y_annot_box2),
            fontsize=12,
            arrowprops=dict(facecolor='#B9B9B9', shrink=0.05))
        
        # Draw hlines 
        day_array = np.arange(end_data+1, forecast_day+1)
        alpha =  np.flip( np.linspace(1, len(day_array), len(day_array)) )
        print(alpha)
        for line, myalpha in zip(day_array, alpha): 
            plt.axvline(x=line, alpha = 1./(myalpha*2), color = color, linestyle = "-")
        
        print()



## if you want to change the country just change the name        
wrap([ "Tunisia", ], data, forecast=5)
plt.savefig("./Tunisia_forcast23March.pdf")