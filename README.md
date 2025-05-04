## Introduction
This project is originally a technical assessment for AI Singapore Apprenticeship programme. The task involves building a machine learning
pipeline for predicting solar panel efficiency based on weather and atmospheric conditions to support optimization of operational
planning for power generation. Details of the task can be found in the PDF file named "AIAP 18 Technical Assessment" above. Datasets can be 
accessed in the "data" folder in "src" directory. Here, a machine learning pipeline based on 3 distinct models are developed that can predict
solar panel efficiency from environmental conditions with good accuracy. 

## Methodology
A machine learning pipleine is built using Random Forest, XgBoost and LightGBM algorithms. As the dataset is imbalanced with respect to the
levels of solar efficiency panel, Balanced Accuracy is used as the primary metrics for performance evalution. Overall AUC is used as a
supplementary metrics.

## Key Results
The key prediction results from the 3 algorithms for the test dataset are as summarized below in Table 1. 

<p align="center"><strong>Table 1: Prediction Performances of 3 Different Models.</strong></p>
<table align="center">
  <tr>
    <th>Model</th>
    <th>Balanced Accuracy</th>
    <th>Overall AUC</th>
  </tr>
  <tr>
    <td>Random Forest (1)</td>
    <td>0.7854</td>
    <td>0.8418</td>
  </tr>
  <tr>
    <td>XgBoost (2)</td>
    <td>0.7875</td>
    <td>0.8500</td>
  </tr>
  <tr>
    <td>LightGBM (3)</td>
    <td>0.7848</td>
    <td>0.8363</td>
</table>

Figure 2 and 3 show the ROC curves and feature ranking results produced from XgBoost.

<p align="center">
  <img src="https://github.com/user-attachments/assets/313063e5-044d-4bbf-b82c-1fb96ecb1787" alt="Diagram" width="350" height='300'/>
</p>
<p align="center"><em>Figure 2: Receiver Operating Curve - Prediction Performance of XgBoost.</em></p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/4be2ef68-aeb9-4476-bae6-45cf732e5579" alt="Diagram" width="1200" height='400'/>
</p>
<p align="center"><em>Figure 3: Ranking of Features in Order of Descending Importance By XgBoost.</em></p>


## Conclusion
Our machine learning analysis shows that Sunshine Duration and Cloud Cover are 2 weather variables that consistently ranked among the top 3 features 
most predictive of (and associated with) solar panel efficiency across the 3 different models. This suggests that knowing these 2 weather conditions
in advanced could help in operational planning.
