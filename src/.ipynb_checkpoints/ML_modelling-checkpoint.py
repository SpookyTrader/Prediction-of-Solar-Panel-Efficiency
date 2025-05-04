#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, balanced_accuracy_score, make_scorer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

def hyperparameter_tuning(X_train, y_train):

    model1 = RandomForestClassifier(class_weight='balanced', random_state=5)

    params1 = {'max_depth': np.arange(3,15,1),
              'criterion':['gini', 'entropy', 'log_loss'],
              'max_features':['sqrt','log2',None],
              'min_samples_split': np.arange(2, 20, 1),
              'n_estimators': np.arange(100,1000,50)}
    
    bal_acc = make_scorer(balanced_accuracy_score)
    
    best_tree1 = RandomizedSearchCV(estimator=model1, param_distributions=params1, scoring=bal_acc, cv=5, n_iter=100, n_jobs=-1, 
                                   verbose=True, random_state=5)
    best_tree1.fit(X_train, y_train)
    
    best_params_rf = best_tree1.best_params_
    best_score_rf = best_tree1.best_score_
    
    print('\nBest parameters for Random Forest:', best_params_rf)
    print('\nBest score for Random forest:', best_score_rf)
    
    class_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model2 = xgb.XGBClassifier(random_state=5, objective='multi:softmax',num_class=3)
    
    params2 = {'max_depth': np.arange(3,15,1),
                  'learning_rate': np.arange(0.01,0.2,0.01),
                  'subsample': np.arange(0.5, 1.0, 0.01),
                  'colsample_bytree': np.arange(0.5, 1.0, 0.01),
                  'colsample_bylevel': np.arange(0.5, 1.0, 0.01),
                  'n_estimators': np.arange(100,1000,50)}
    
    best_tree2 = RandomizedSearchCV(estimator=model2, param_distributions=params2, scoring=bal_acc, cv=5, n_iter=100, n_jobs=-1, 
                                   verbose=True, random_state=5)
    best_tree2.fit(X_train, y_train, sample_weight=class_weights)

    best_params_xgb = best_tree2.best_params_
    best_score_xgb = best_tree2.best_score_
    
    print('\nBest parameters for XgBoost:', best_params_xgb)
    print('\nBest score for XgBoost:', best_score_xgb)

    model3 = LGBMClassifier(objective='multiclass', class_weight='balanced', num_class=3, verbose=-1, random_state=5)

    params3 = {'max_depth': np.arange(3,15,1),
              'learning_rate': np.arange(0.01,0.2,0.01),
              'subsample': np.arange(0.2, 1.0, 0.01),
              'n_estimators': np.arange(100,1000,50)}
    
    best_tree3 = RandomizedSearchCV(estimator=model3, param_distributions=params3, scoring=bal_acc, cv=5, n_iter=100, n_jobs=-1, 
                                   verbose=True, random_state=5)
    best_tree3.fit(X_train, y_train, sample_weight=class_weights)
    
    best_params_lgbm = best_tree3.best_params_
    best_score_lgbm = best_tree3.best_score_
    
    print('\nBest parameters for LightGBM:', best_params_lgbm)
    print('\nBest score for LightGBM:', best_score_lgbm)
    
    return best_params_rf, best_params_xgb, best_params_lgbm, model1, model2, model3

def prediction_results(X_train, X_test, y_train, y_test, best_params_rf, best_params_xgb, best_params_lgbm, model1, model2, model3, charts=False):
    i = 0
    parameters = [best_params_rf, best_params_xgb, best_params_lgbm]
    for m in [model1,model2,model3]:
        i+=1
        m.set_params(**parameters[i-1])
    
        m.fit(X_train, y_train)
    
        print(f'Results for Model {i}:')
        
        y_predict = m.predict(X_test)
        print('Prediction on test data:', y_predict)
        
        y_predict_proba = m.predict_proba(X_test)
        print('\nPredicted probabilities:',y_predict_proba)

        confusion = metrics.confusion_matrix(y_test, y_predict)
        print('\nConfusion matrix:\n', confusion)
        if charts:
            confusion_df = pd.DataFrame(confusion, columns=np.unique(y_test), index = np.unique(y_test))
            plt.figure(figsize = (3,3))
            plt.rcParams.update({'font.size': 15})
            sns.heatmap(confusion_df, cmap = 'Blues', annot=True, fmt='g', square=True, linewidths=.5, cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Expected')
            plt.tight_layout()
            plt.savefig('confusion_matrix_model'+str(i)+'.png')
            plt.close()
        
        bal_accuracy = balanced_accuracy_score(y_test,y_predict)
        print('\nBalanced_Accuracy_score on test dataset: ', bal_accuracy)

        print('\nOther Scores:\n', classification_report(y_test, y_predict))

        overallAUC = roc_auc_score(y_test, y_predict_proba, multi_class='ovr')
        print('AUC:',overallAUC)
        if charts:
            colors = ['blue', 'red', 'green']
            eff = ['Low', 'Medium', 'High']
            for n in range(len(eff)):
                
                y_true = y_test.values.tolist()
                
                for j in range(len(y_true)):
                    if y_true[j] == n:
                        y_true[j]=1
                    else:
                        y_true[j]=0
                        
                fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba[:,n].tolist())
                plt.plot(fpr, tpr, color=colors[n], lw=2, label=f'Class {eff[n]} (AUC={auc(fpr, tpr):.5f})')
            
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Multiclass ROC Curve - One vs Rest\n(Overall AUC={overallAUC:.5f})')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig('ROC_model'+str(i)+'.png')
            plt.close()

        fsi = pd.Series(m.feature_importances_, index=X_train.columns)
        fsi_sorted = (fsi/fsi.sum()).sort_values(ascending=False)
        if charts:
            plt.figure(figsize=(15, 5))
            plt.bar(fsi_sorted.index, fsi_sorted.values, color='blue')
            plt.xlabel('Features')
            plt.ylabel('Relative Importance')
            plt.xticks(rotation=90, fontsize=10)
            plt.yticks(fontsize=10)
            plt.title("Ranking of feature's importance")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('feature_ranking_model'+str(i)+'.png')
            plt.close()
        fsi_df = fsi_sorted.to_frame().reset_index().rename(columns={'index':'Feature',0:'Relative Importance'})
        fsi_df.index += 1
        print(f'\nFeatures ranked by importance:\n{fsi_df}')
        
    return None