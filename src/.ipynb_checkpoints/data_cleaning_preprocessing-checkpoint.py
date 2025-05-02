#!/usr/bin/env python

import sqlite3
import pandas as pd
import numpy as np

def data_cleaning_processing():
    
    def rename_wind_dir(x):
        if x in ['N','N.','north','NORTH','Northward']:
            return 'North'
        elif x in ['S','S.','south','SOUTH','Southward']:
            return 'South'
        elif x in ['E','E.','east','EAST']:
            return 'East'
        elif x in ['W','W.','west','WEST']:
            return 'West'
        elif x in ['NE','NE.','northeast','NORTHEAST']:
            return 'NorthEast'
        elif x in ['SE','SE.','southeast','SOUTHEAST']:
            return 'SouthEast'
        elif x in ['SW','SW.']:
            return 'SouthWest'
        elif x in ['NW','NW.','northwest','NORTHWEST']:
            return 'NorthWest'
        else:
            return x

    def rename_dew_pt(x):
        if x in ['Very High','High','High Level','H','HIGH','high','very high','VERY HIGH','VH','Extreme']:
            return 'High'
        elif x in ['moderate','Normal','MODERATE','M','Moderate']:
            return 'Moderate'
        elif x in ['Very Low','Low','LOW','VL','very low','low','L','VERY LOW','Below Average','Minimal']:
            return 'Low'
    
    def cross_table(series1, series2):
        xtable = pd.crosstab(series1, series2)
        cols = xtable.columns.to_list()
        
        xtable['Total'] = xtable[cols].sum(axis=1)
        
        for f in cols:
            xtable[f+'(%)'] = (xtable[f]/xtable['Total'])*100
    
        return xtable

    con = sqlite3.connect("src/data/air_quality.db")
    air = pd.read_sql_query("SELECT * from air_quality", con)
    con = sqlite3.connect("src/data/weather.db")
    weather = pd.read_sql_query("SELECT * from weather", con)
    print('extracting data...')
    print(air.head())
    print('\n\n',weather.head())

    air['date'] = pd.to_datetime(air['date'], dayfirst=True)
    weather['date'] = pd.to_datetime(weather['date'], dayfirst=True)

    air.drop('data_ref', axis=1, inplace=True)
    weather.drop('data_ref', axis=1, inplace=True)

    air.drop_duplicates(keep='first',ignore_index=True, inplace=True)
    air = air.set_index('date').stack().unstack()
    air.reset_index(inplace=True)

    weather.drop_duplicates(keep='first',ignore_index=True, inplace=True)

    merged = air.merge(weather, how='inner', on='date')
    print('\n\nMerged dataframe:\n')
    print(merged.head())

    for c in merged.columns[1:24]:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')

    merged.fillna(merged.select_dtypes('float').median(), inplace=True)

    merged['Dew Point Category'] = merged['Dew Point Category'].apply(rename_dew_pt)
    merged['Wind Direction'] = merged['Wind Direction'].apply(rename_wind_dir)

    merged = merged.drop(merged.loc[merged['Max Wind Speed (km/h)']<0].index)
    merged = merged.drop(merged.loc[merged['Wet Bulb Temperature (deg F)']<0].index)
    merged.reset_index(drop=True, inplace=True)

    merged['pm25_average'] = merged[['pm25_north','pm25_south','pm25_east','pm25_west','pm25_central']].mean(axis=1)
    merged['psi_average'] = merged[['psi_north','psi_south','psi_east','psi_west','psi_central']].mean(axis=1)
    merged['temp_range'] = merged['Maximum Temperature (deg C)']-merged['Min Temperature (deg C)']
    merged['wind_range'] = merged['Max Wind Speed (km/h)']-merged['Min Wind Speed (km/h)']

    print('\n\nCleaned and processed merged dataset:\n')
    merged.info()

    print('\n\nDone!!!')

    return merged









