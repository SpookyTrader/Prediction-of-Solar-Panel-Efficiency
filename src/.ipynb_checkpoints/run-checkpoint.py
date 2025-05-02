#!/usr/bin/env python

from data_cleaning_preprocessing import data_cleaning_processing
from data_prep_for_ML import train_test_data
from ML_modelling import hyperparameter_tuning, prediction_results

df = data_cleaning_processing()

X_train, X_test, y_train, y_test = train_test_data(df)

best_params_rf, best_params_xgb, best_params_lgbm, model1, model2, model3 = hyperparameter_tuning(X_train, y_train)

prediction_results(X_train, X_test, y_train, y_test, best_params_rf, best_params_xgb, best_params_lgbm, model1, model2, model3, charts=True)