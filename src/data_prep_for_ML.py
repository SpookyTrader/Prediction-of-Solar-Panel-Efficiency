#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def train_test_data(df):

    print('Processing in progress.....')
    
    def efficiency_labelling(efficiency, efficiency_labels):
        return efficiency_labels[efficiency]

    efficiency_labels = {'Low':0, 'Medium':1, 'High':2}
    df['Daily Solar Panel Efficiency'] = df['Daily Solar Panel Efficiency'].apply(efficiency_labelling, efficiency_labels=efficiency_labels)

    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    X_encoded = onehot_encoder.fit_transform(df[['Dew Point Category', 'Wind Direction']])
    X_encoded_df = pd.DataFrame(X_encoded)
    X_encoded_df.columns = onehot_encoder.get_feature_names_out()
    df = pd.concat([df, X_encoded_df], axis=1)

    X = df.drop(['date','Dew Point Category','Wind Direction','Daily Solar Panel Efficiency'], axis=1)
    y = df['Daily Solar Panel Efficiency']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.30, random_state=5)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    print('Completed!!!')
    
    return X_train, X_test, y_train, y_test