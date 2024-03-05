***
# Supervised Learning : Comparing Batch, Stochastic and Mini-Batch Approaches to Gradient Descent in Estimating Regression Coefficients

***
### John Pauline Pineda <br> <br> *March 10, 2024*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Linear Regression Model Coefficient Estimation](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Normal Equations](#1.6.2)
        * [1.6.3 Gradient Descent Algorithm with Very High Learning Rate and Low Epoch Count ](#1.6.3)
        * [1.6.4 Gradient Descent Algorithm with Very High Learning Rate and High Epoch Count](#1.6.4)
        * [1.6.5 Gradient Descent Algorithm with High Learning Rate and Low Epoch Count](#1.6.5)
        * [1.6.6 Gradient Descent Algorithm with High Learning Rate and High Epoch Count](#1.6.6)
        * [1.6.7 Gradient Descent Algorithm with Low Learning Rate and Low Epoch Count](#1.6.7)
        * [1.6.8 Gradient Descent Algorithm with Low Learning Rate and High Epoch Count](#1.6.8)
    * [1.7 Consolidated Findings](#1.7)
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project manually implements the **Gradient Descent** algorithm using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark>, and evaluates a range of values for the learning rate and epoch count parameters to optimally estimate the coefficients of a linear regression model. The cost function optimization profiles of the different candidate parameter settings were compared, with the resulting estimated coefficients assessed against those obtained using normal equations which served as the reference baseline values. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document.

[Linear Regression](https://link.springer.com/book/10.1007/978-1-4757-3462-1) explores the linear relationship between a scalar response and one or more covariates by having the conditional mean of the dependent variable be an affine function of the independent variables. The relationship is modeled through a disturbance term which represents an unobserved random variable that adds noise. The algorithm is typically formulated from the data using the least squares method which seeks to estimate the coefficients by minimizing the squared residual function. The linear equation assigns one scale factor represented by a coefficient to each covariate and an additional coefficient called the intercept or the bias coefficient which gives the line an additional degree of freedom allowing to move up and down a two-dimensional plot.

[Regression coefficients](https://link.springer.com/book/10.1007/978-1-4757-3462-1) represent the changes in the independent variable which explain the variation of the dependent variable in the model. The methods applied in this study attempt to estimate the unknown model coefficients by optimizing a loss function - that which measures the quality of the estimated parameters based on how well the model-induced scores agree with the ground truth labels in the data set.

[Normal Equations](https://www.semanticscholar.org/paper/Nouvelles-m%C3%A9thodes-pour-la-d%C3%A9termination-des-des-Legendre/21c9090e226ab449ffb608ddb2cb925911a61f24) are a system of equations whose solution is the Ordinary Least Squares (OLS) estimator of the regression coefficients and which are derived from the first-order condition of the least squares minimization problem. These equations are obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares). This approach is a closed-form solution and a one-step algorithm used to analytically find the coefficients that minimize the loss function.

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. 

## 1.1. Data Background <a class="anchor" id="1.1"></a>

Datasets used for the analysis were separately gathered and consolidated from various sources including: 
1. Cancer Rates from [World Population Review](https://worldpopulationreview.com/country-rankings/cancer-rates-by-country)
2. Social Protection and Labor Indicator from [World Bank](https://data.worldbank.org/topic/social-protection-and-labor?view=chart)
3. Education Indicator from [World Bank](https://data.worldbank.org/topic/education?view=chart)
4. Economy and Growth Indicator from [World Bank](https://data.worldbank.org/topic/economy-and-growth?view=chart)
5. Environment Indicator from [World Bank](https://data.worldbank.org/topic/environment?view=chart)
6. Climate Change Indicator from [World Bank](https://data.worldbank.org/topic/climate-change?view=chart)
7. Agricultural and Rural Development Indicator from [World Bank](https://data.worldbank.org/topic/agriculture-and-rural-development?view=chart)
8. Social Development Indicator from [World Bank](https://data.worldbank.org/topic/social-development?view=chart)
9. Health Indicator from [World Bank](https://data.worldbank.org/topic/health?view=chart)
10. Science and Technology Indicator from [World Bank](https://data.worldbank.org/topic/science-and-technology?view=chart)
11. Urban Development Indicator from [World Bank](https://data.worldbank.org/topic/urban-development?view=chart)
12. Human Development Indices from [Human Development Reports](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)
13. Environmental Performance Indices from [Yale Center for Environmental Law and Policy](https://epi.yale.edu/epi-results/2022/component/epi)

This study hypothesized that various global development indicators and indices influence cancer rates across countries.

The target variable for the study is:
* <span style="color: #FF0000">CANRAT</span> - Age-standardized cancer rates, per 100K population (2022)

The predictor variables for the study are:
* <span style="color: #FF0000">GDPPER</span> - GDP per person employed, current US Dollars (2020)
* <span style="color: #FF0000">URBPOP</span> - Urban population, % of total population (2020)
* <span style="color: #FF0000">PATRES</span> - Patent applications by residents, total count (2020)
* <span style="color: #FF0000">RNDGDP</span> - Research and development expenditure, % of GDP (2020)
* <span style="color: #FF0000">POPGRO</span> - Population growth, annual % (2020)
* <span style="color: #FF0000">LIFEXP</span> - Life expectancy at birth, total in years (2020)
* <span style="color: #FF0000">TUBINC</span> - Incidence of tuberculosis, per 100K population (2020)
* <span style="color: #FF0000">DTHCMD</span> - Cause of death by communicable diseases and maternal, prenatal and nutrition conditions,  % of total (2019)
* <span style="color: #FF0000">AGRLND</span> - Agricultural land,  % of land area (2020)
* <span style="color: #FF0000">GHGEMI</span> - Total greenhouse gas emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">RELOUT</span> - Renewable electricity output, % of total electricity output (2015)
* <span style="color: #FF0000">METEMI</span> - Methane emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">FORARE</span> - Forest area, % of land area (2020)
* <span style="color: #FF0000">CO2EMI</span> - CO2 emissions, metric tons per capita (2020)
* <span style="color: #FF0000">PM2EXP</span> - PM2.5 air pollution, population exposed to levels exceeding WHO guideline value,  % of total (2017)
* <span style="color: #FF0000">POPDEN</span> - Population density, people per sq. km of land area (2020)
* <span style="color: #FF0000">GDPCAP</span> - GDP per capita, current US Dollars (2020)
* <span style="color: #FF0000">ENRTER</span> - Tertiary school enrollment, % gross (2020)
* <span style="color: #FF0000">HDICAT</span> - Human development index, ordered category (2020)
* <span style="color: #FF0000">EPISCO</span> - Environment performance index , score (2022)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **177 rows** (observations)
    * **22 columns** (variables)
        * **1/22 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/22 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **19/22 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">PATRES</span>
             * <span style="color: #FF0000">RNDGDP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">RELOUT</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">ENRTER</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/22 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy import stats

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline
```


```python
##################################
# Loading the dataset
##################################
cancer_rate = pd.read_csv('NumericCancerRates.csv')
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate.dtypes)
```

    Column Names and Data Types:
    


    COUNTRY     object
    CANRAT     float64
    GDPPER     float64
    URBPOP     float64
    PATRES     float64
    RNDGDP     float64
    POPGRO     float64
    LIFEXP     float64
    TUBINC     float64
    DTHCMD     float64
    AGRLND     float64
    GHGEMI     float64
    RELOUT     float64
    METEMI     float64
    FORARE     float64
    CO2EMI     float64
    PM2EXP     float64
    POPDEN     float64
    ENRTER     float64
    GDPCAP     float64
    HDICAT      object
    EPISCO     float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COUNTRY</th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>PATRES</th>
      <th>RNDGDP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>...</th>
      <th>RELOUT</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>ENRTER</th>
      <th>GDPCAP</th>
      <th>HDICAT</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>452.4</td>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>2368.0</td>
      <td>NaN</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>...</td>
      <td>13.637841</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>110.139221</td>
      <td>51722.06900</td>
      <td>VH</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>422.9</td>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>348.0</td>
      <td>NaN</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>...</td>
      <td>80.081439</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>75.734833</td>
      <td>41760.59478</td>
      <td>VH</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>372.8</td>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>75.0</td>
      <td>1.23244</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>...</td>
      <td>27.965408</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>74.680313</td>
      <td>85420.19086</td>
      <td>VH</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>362.2</td>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>269586.0</td>
      <td>3.42287</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>...</td>
      <td>13.228593</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>87.567657</td>
      <td>63528.63430</td>
      <td>VH</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>351.1</td>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>1261.0</td>
      <td>2.96873</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>...</td>
      <td>65.505925</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>82.664330</td>
      <td>60915.42440</td>
      <td>VH</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Setting the levels of the categorical variables
##################################
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].astype('category')
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].cat.set_categories(['L', 'M', 'H', 'VH'], ordered=True)
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cancer_rate.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT</th>
      <td>177.0</td>
      <td>183.829379</td>
      <td>7.974340e+01</td>
      <td>78.400000</td>
      <td>118.100000</td>
      <td>155.300000</td>
      <td>240.400000</td>
      <td>4.524000e+02</td>
    </tr>
    <tr>
      <th>GDPPER</th>
      <td>165.0</td>
      <td>45284.424283</td>
      <td>3.941794e+04</td>
      <td>1718.804896</td>
      <td>13545.254510</td>
      <td>34024.900890</td>
      <td>66778.416050</td>
      <td>2.346469e+05</td>
    </tr>
    <tr>
      <th>URBPOP</th>
      <td>174.0</td>
      <td>59.788121</td>
      <td>2.280640e+01</td>
      <td>13.345000</td>
      <td>42.432750</td>
      <td>61.701500</td>
      <td>79.186500</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>PATRES</th>
      <td>108.0</td>
      <td>20607.388889</td>
      <td>1.340683e+05</td>
      <td>1.000000</td>
      <td>35.250000</td>
      <td>244.500000</td>
      <td>1297.750000</td>
      <td>1.344817e+06</td>
    </tr>
    <tr>
      <th>RNDGDP</th>
      <td>74.0</td>
      <td>1.197474</td>
      <td>1.189956e+00</td>
      <td>0.039770</td>
      <td>0.256372</td>
      <td>0.873660</td>
      <td>1.608842</td>
      <td>5.354510e+00</td>
    </tr>
    <tr>
      <th>POPGRO</th>
      <td>174.0</td>
      <td>1.127028</td>
      <td>1.197718e+00</td>
      <td>-2.079337</td>
      <td>0.236900</td>
      <td>1.179959</td>
      <td>2.031154</td>
      <td>3.727101e+00</td>
    </tr>
    <tr>
      <th>LIFEXP</th>
      <td>174.0</td>
      <td>71.746113</td>
      <td>7.606209e+00</td>
      <td>52.777000</td>
      <td>65.907500</td>
      <td>72.464610</td>
      <td>77.523500</td>
      <td>8.456000e+01</td>
    </tr>
    <tr>
      <th>TUBINC</th>
      <td>174.0</td>
      <td>105.005862</td>
      <td>1.367229e+02</td>
      <td>0.770000</td>
      <td>12.000000</td>
      <td>44.500000</td>
      <td>147.750000</td>
      <td>5.920000e+02</td>
    </tr>
    <tr>
      <th>DTHCMD</th>
      <td>170.0</td>
      <td>21.260521</td>
      <td>1.927333e+01</td>
      <td>1.283611</td>
      <td>6.078009</td>
      <td>12.456279</td>
      <td>36.980457</td>
      <td>6.520789e+01</td>
    </tr>
    <tr>
      <th>AGRLND</th>
      <td>174.0</td>
      <td>38.793456</td>
      <td>2.171551e+01</td>
      <td>0.512821</td>
      <td>20.130276</td>
      <td>40.386649</td>
      <td>54.013754</td>
      <td>8.084112e+01</td>
    </tr>
    <tr>
      <th>GHGEMI</th>
      <td>170.0</td>
      <td>259582.709895</td>
      <td>1.118550e+06</td>
      <td>179.725150</td>
      <td>12527.487367</td>
      <td>41009.275980</td>
      <td>116482.578575</td>
      <td>1.294287e+07</td>
    </tr>
    <tr>
      <th>RELOUT</th>
      <td>153.0</td>
      <td>39.760036</td>
      <td>3.191492e+01</td>
      <td>0.000296</td>
      <td>10.582691</td>
      <td>32.381668</td>
      <td>63.011450</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>METEMI</th>
      <td>170.0</td>
      <td>47876.133575</td>
      <td>1.346611e+05</td>
      <td>11.596147</td>
      <td>3662.884908</td>
      <td>11118.976025</td>
      <td>32368.909040</td>
      <td>1.186285e+06</td>
    </tr>
    <tr>
      <th>FORARE</th>
      <td>173.0</td>
      <td>32.218177</td>
      <td>2.312001e+01</td>
      <td>0.008078</td>
      <td>11.604388</td>
      <td>31.509048</td>
      <td>49.071780</td>
      <td>9.741212e+01</td>
    </tr>
    <tr>
      <th>CO2EMI</th>
      <td>170.0</td>
      <td>3.751097</td>
      <td>4.606479e+00</td>
      <td>0.032585</td>
      <td>0.631924</td>
      <td>2.298368</td>
      <td>4.823496</td>
      <td>3.172684e+01</td>
    </tr>
    <tr>
      <th>PM2EXP</th>
      <td>167.0</td>
      <td>91.940595</td>
      <td>2.206003e+01</td>
      <td>0.274092</td>
      <td>99.627134</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
    </tr>
    <tr>
      <th>POPDEN</th>
      <td>174.0</td>
      <td>200.886765</td>
      <td>6.453834e+02</td>
      <td>2.115134</td>
      <td>27.454539</td>
      <td>77.983133</td>
      <td>153.993650</td>
      <td>7.918951e+03</td>
    </tr>
    <tr>
      <th>ENRTER</th>
      <td>116.0</td>
      <td>49.994997</td>
      <td>2.970619e+01</td>
      <td>2.432581</td>
      <td>22.107195</td>
      <td>53.392460</td>
      <td>71.057467</td>
      <td>1.433107e+02</td>
    </tr>
    <tr>
      <th>GDPCAP</th>
      <td>170.0</td>
      <td>13992.095610</td>
      <td>1.957954e+04</td>
      <td>216.827417</td>
      <td>1870.503029</td>
      <td>5348.192875</td>
      <td>17421.116227</td>
      <td>1.173705e+05</td>
    </tr>
    <tr>
      <th>EPISCO</th>
      <td>165.0</td>
      <td>42.946667</td>
      <td>1.249086e+01</td>
      <td>18.900000</td>
      <td>33.000000</td>
      <td>40.900000</td>
      <td>50.500000</td>
      <td>7.790000e+01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variable
##################################
print('Object Variable Summary:')
display(cancer_rate.describe(include='object').transpose())
```

    Object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COUNTRY</th>
      <td>177</td>
      <td>177</td>
      <td>Australia</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variable
##################################
print('Categorical Variable Summary:')
display(cancer_rate.describe(include='category').transpose())
```

    Categorical Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HDICAT</th>
      <td>167</td>
      <td>4</td>
      <td>VH</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 20 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
    * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
    * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
    * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">EPISCO</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">HDICAT</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">DTHCMD</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">METEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">CO2EMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GDPCAP</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GHGEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 4, Fill.Rate = 0.977
    * <span style="color: #FF0000">TUBINC</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">AGRLND</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPGRO</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPDEN</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">URBPOP</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">LIFEXP</span>: Null.Count = 3, Fill.Rate = 0.983
3. 120 observations noted with at least 1 missing data. From this number, 14 observations reported high Missing.Rate>0.2.
    * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
    * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227
4. Low variance observed for 1 variable with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">PM2EXP</span>: First.Second.Mode.Ratio = 53.000
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High skewness observed for 5 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">POPDEN</span>: Skewness = +10.267
    * <span style="color: #FF0000">GHGEMI</span>: Skewness = +9.496
    * <span style="color: #FF0000">PATRES</span>: Skewness = +9.284
    * <span style="color: #FF0000">METEMI</span>: Skewness = +5.801
    * <span style="color: #FF0000">PM2EXP</span>: Skewness = -3.141


```python
##################################
# Counting the number of duplicated rows
##################################
cancer_rate.duplicated().sum()
```




    0




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate)] * len(cancer_rate.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>float64</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.610169</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.418079</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.864407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.977401</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.655367</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])
```




    20




```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.418079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.610169</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.655367</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.864407</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.932203</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.943503</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.960452</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.977401</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.983051</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 0.90
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<0.90)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cancer_rate["COUNTRY"].values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cancer_rate.columns)] * len(cancer_rate))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cancer_rate.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Identifying the rows
# with missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22</td>
      <td>1</td>
      <td>0.045455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>22</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Congo Republic</td>
      <td>22</td>
      <td>3</td>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Bhutan</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Nepal</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Gambia</td>
      <td>22</td>
      <td>4</td>
      <td>0.181818</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Niger</td>
      <td>22</td>
      <td>2</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.00
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.00)])
```




    120




```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.20
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)])
```




    14




```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)]
```


```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
display(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)].sort_values(by=['Missing.Rate'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>Guadeloupe</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Martinique</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>56</th>
      <td>French Guiana</td>
      <td>22</td>
      <td>20</td>
      <td>0.909091</td>
    </tr>
    <tr>
      <th>13</th>
      <td>New Caledonia</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>French Polynesia</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Guam</td>
      <td>22</td>
      <td>11</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Puerto Rico</td>
      <td>22</td>
      <td>9</td>
      <td>0.409091</td>
    </tr>
    <tr>
      <th>85</th>
      <td>North Korea</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Somalia</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>168</th>
      <td>South Sudan</td>
      <td>22</td>
      <td>6</td>
      <td>0.272727</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Venezuela</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Libya</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Eritrea</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Yemen</td>
      <td>22</td>
      <td>5</td>
      <td>0.227273</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cancer_rate_numeric = cancer_rate.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cancer_rate_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cancer_rate_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cancer_rate_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cancer_rate_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cancer_rate_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[0] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[1] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cancer_rate_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cancer_rate_numeric)] * len(cancer_rate_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cancer_rate_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>78.400000</td>
      <td>183.829379</td>
      <td>155.300000</td>
      <td>4.524000e+02</td>
      <td>135.300000</td>
      <td>106.700000</td>
      <td>3</td>
      <td>2</td>
      <td>1.500000</td>
      <td>167</td>
      <td>177</td>
      <td>0.943503</td>
      <td>0.881825</td>
      <td>0.063467</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDPPER</td>
      <td>1718.804896</td>
      <td>45284.424283</td>
      <td>34024.900890</td>
      <td>2.346469e+05</td>
      <td>98380.636010</td>
      <td>42154.178100</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>165</td>
      <td>177</td>
      <td>0.932203</td>
      <td>1.517574</td>
      <td>3.471992</td>
    </tr>
    <tr>
      <th>2</th>
      <td>URBPOP</td>
      <td>13.345000</td>
      <td>59.788121</td>
      <td>61.701500</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>52.516000</td>
      <td>2</td>
      <td>1</td>
      <td>2.000000</td>
      <td>173</td>
      <td>177</td>
      <td>0.977401</td>
      <td>-0.210702</td>
      <td>-0.962847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PATRES</td>
      <td>1.000000</td>
      <td>20607.388889</td>
      <td>244.500000</td>
      <td>1.344817e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>97</td>
      <td>177</td>
      <td>0.548023</td>
      <td>9.284436</td>
      <td>91.187178</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RNDGDP</td>
      <td>0.039770</td>
      <td>1.197474</td>
      <td>0.873660</td>
      <td>5.354510e+00</td>
      <td>1.232440</td>
      <td>0.962180</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>74</td>
      <td>177</td>
      <td>0.418079</td>
      <td>1.396742</td>
      <td>1.695957</td>
    </tr>
    <tr>
      <th>5</th>
      <td>POPGRO</td>
      <td>-2.079337</td>
      <td>1.127028</td>
      <td>1.179959</td>
      <td>3.727101e+00</td>
      <td>1.235701</td>
      <td>1.483129</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>-0.195161</td>
      <td>-0.423580</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LIFEXP</td>
      <td>52.777000</td>
      <td>71.746113</td>
      <td>72.464610</td>
      <td>8.456000e+01</td>
      <td>83.200000</td>
      <td>68.687000</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>-0.357965</td>
      <td>-0.649601</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TUBINC</td>
      <td>0.770000</td>
      <td>105.005862</td>
      <td>44.500000</td>
      <td>5.920000e+02</td>
      <td>12.000000</td>
      <td>7.200000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>131</td>
      <td>177</td>
      <td>0.740113</td>
      <td>1.746333</td>
      <td>2.429368</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DTHCMD</td>
      <td>1.283611</td>
      <td>21.260521</td>
      <td>12.456279</td>
      <td>6.520789e+01</td>
      <td>4.941054</td>
      <td>42.079403</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>0.900509</td>
      <td>-0.691541</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AGRLND</td>
      <td>0.512821</td>
      <td>38.793456</td>
      <td>40.386649</td>
      <td>8.084112e+01</td>
      <td>46.252480</td>
      <td>72.006469</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>0.074000</td>
      <td>-0.926249</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GHGEMI</td>
      <td>179.725150</td>
      <td>259582.709895</td>
      <td>41009.275980</td>
      <td>1.294287e+07</td>
      <td>571903.119900</td>
      <td>3000.932259</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>9.496120</td>
      <td>101.637308</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RELOUT</td>
      <td>0.000296</td>
      <td>39.760036</td>
      <td>32.381668</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>13.637841</td>
      <td>3</td>
      <td>1</td>
      <td>3.000000</td>
      <td>151</td>
      <td>177</td>
      <td>0.853107</td>
      <td>0.501088</td>
      <td>-0.981774</td>
    </tr>
    <tr>
      <th>12</th>
      <td>METEMI</td>
      <td>11.596147</td>
      <td>47876.133575</td>
      <td>11118.976025</td>
      <td>1.186285e+06</td>
      <td>131484.763200</td>
      <td>1326.034028</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>5.801014</td>
      <td>38.661386</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FORARE</td>
      <td>0.008078</td>
      <td>32.218177</td>
      <td>31.509048</td>
      <td>9.741212e+01</td>
      <td>17.421315</td>
      <td>8.782159</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>173</td>
      <td>177</td>
      <td>0.977401</td>
      <td>0.519277</td>
      <td>-0.322589</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CO2EMI</td>
      <td>0.032585</td>
      <td>3.751097</td>
      <td>2.298368</td>
      <td>3.172684e+01</td>
      <td>14.772658</td>
      <td>0.972088</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>2.721552</td>
      <td>10.311574</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>106</td>
      <td>2</td>
      <td>53.000000</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
    <tr>
      <th>16</th>
      <td>POPDEN</td>
      <td>2.115134</td>
      <td>200.886765</td>
      <td>77.983133</td>
      <td>7.918951e+03</td>
      <td>3.335312</td>
      <td>13.300785</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>10.267750</td>
      <td>119.995256</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ENRTER</td>
      <td>2.432581</td>
      <td>49.994997</td>
      <td>53.392460</td>
      <td>1.433107e+02</td>
      <td>110.139221</td>
      <td>45.220661</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>116</td>
      <td>177</td>
      <td>0.655367</td>
      <td>0.275863</td>
      <td>-0.392895</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GDPCAP</td>
      <td>216.827417</td>
      <td>13992.095610</td>
      <td>5348.192875</td>
      <td>1.173705e+05</td>
      <td>51722.069000</td>
      <td>3961.726633</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>2.258568</td>
      <td>5.938690</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EPISCO</td>
      <td>18.900000</td>
      <td>42.946667</td>
      <td>40.900000</td>
      <td>7.790000e+01</td>
      <td>29.600000</td>
      <td>43.600000</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
      <td>137</td>
      <td>177</td>
      <td>0.774011</td>
      <td>0.641799</td>
      <td>0.035208</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    1




```python
##################################
# Identifying the numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>106</td>
      <td>2</td>
      <td>53.0</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```




    5




```python
##################################
# Identifying the numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))].sort_values(by=['Skewness'], ascending=False))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>POPDEN</td>
      <td>2.115134</td>
      <td>200.886765</td>
      <td>77.983133</td>
      <td>7.918951e+03</td>
      <td>3.335312</td>
      <td>13.300785</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>174</td>
      <td>177</td>
      <td>0.983051</td>
      <td>10.267750</td>
      <td>119.995256</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GHGEMI</td>
      <td>179.725150</td>
      <td>259582.709895</td>
      <td>41009.275980</td>
      <td>1.294287e+07</td>
      <td>571903.119900</td>
      <td>3000.932259</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>9.496120</td>
      <td>101.637308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PATRES</td>
      <td>1.000000</td>
      <td>20607.388889</td>
      <td>244.500000</td>
      <td>1.344817e+06</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>4</td>
      <td>3</td>
      <td>1.333333</td>
      <td>97</td>
      <td>177</td>
      <td>0.548023</td>
      <td>9.284436</td>
      <td>91.187178</td>
    </tr>
    <tr>
      <th>12</th>
      <td>METEMI</td>
      <td>11.596147</td>
      <td>47876.133575</td>
      <td>11118.976025</td>
      <td>1.186285e+06</td>
      <td>131484.763200</td>
      <td>1326.034028</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>170</td>
      <td>177</td>
      <td>0.960452</td>
      <td>5.801014</td>
      <td>38.661386</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PM2EXP</td>
      <td>0.274092</td>
      <td>91.940595</td>
      <td>100.000000</td>
      <td>1.000000e+02</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>106</td>
      <td>2</td>
      <td>53.000000</td>
      <td>61</td>
      <td>177</td>
      <td>0.344633</td>
      <td>-3.141557</td>
      <td>9.032386</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cancer_rate_object = cancer_rate.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cancer_rate_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cancer_rate_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cancer_rate_object)] * len(cancer_rate_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>Australia</td>
      <td>Mauritius</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>177</td>
      <td>177</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Formulating the dataset
# with categorical columns only
##################################
cancer_rate_categorical = cancer_rate.select_dtypes(include='category')
```


```python
##################################
# Gathering the variable names for the categorical column
##################################
categorical_variable_name_list = cancer_rate_categorical.columns
```


```python
##################################
# Gathering the first mode values for each categorical column
##################################
categorical_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the second mode values for each categorical column
##################################
categorical_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of first mode values for each categorical column
##################################
categorical_first_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of second mode values for each categorical column
##################################
categorical_second_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the first mode to second mode ratio for each categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each categorical column
##################################
categorical_unique_count_list = cancer_rate_categorical.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each categorical column
##################################
categorical_row_count_list = list([len(cancer_rate_categorical)] * len(cancer_rate_categorical.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)
```


```python
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                    categorical_first_mode_list,
                                                    categorical_second_mode_list,
                                                    categorical_first_mode_count_list,
                                                    categorical_second_mode_count_list,
                                                    categorical_first_second_mode_ratio_list,
                                                    categorical_unique_count_list,
                                                    categorical_row_count_list,
                                                    categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HDICAT</td>
      <td>VH</td>
      <td>H</td>
      <td>59</td>
      <td>39</td>
      <td>1.512821</td>
      <td>4</td>
      <td>177</td>
      <td>0.022599</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>


### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

1. Subsets of rows and columns with high rates of missing data were removed from the dataset:
    * 4 variables with Fill.Rate<0.9 were excluded for subsequent analysis.
        * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
        * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
        * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
        * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * 14 rows with Missing.Rate>0.2 were exluded for subsequent analysis.
        * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
        * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227  
2. No variables were removed due to zero or near-zero variance.
3. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **15/18 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.20
##################################
cancer_rate_filtered_row = cancer_rate.drop(cancer_rate[cancer_rate.COUNTRY.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (163, 22)



```python
##################################
# Filtering out the columns with
# with Fill.Rate < 0.90
##################################
cancer_rate_filtered_row_column = cancer_rate_filtered_row.drop(column_low_fill_rate['Column.Name'].values.tolist(), axis=1)
```


```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cancer_rate_cleaned = cancer_rate_filtered_row_column
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_cleaned.shape)
```

    Dataset Dimensions: 
    


    (163, 18)


### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

[Iterative Imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) is based on the [Multivariate Imputation by Chained Equations](https://journals.sagepub.com/doi/10.1177/0962280206074463) (MICE) algorithm - an imputation method based on fully conditional specification, where each incomplete variable is imputed by a separate model. As a sequential regression imputation technique, the algorithm imputes an incomplete column (target column) by generating plausible synthetic values given other columns in the data. Each incomplete column must act as a target column, and has its own specific set of predictors. For predictors that are incomplete themselves, the most recently generated imputations are used to complete the predictors prior to prior to imputation of the target columns.

[Linear Regression](https://link.springer.com/book/10.1007/978-1-4757-3462-1) explores the linear relationship between a scalar response and one or more covariates by having the conditional mean of the dependent variable be an affine function of the independent variables. The relationship is modeled through a disturbance term which represents an unobserved random variable that adds noise. The algorithm is typically formulated from the data using the least squares method which seeks to estimate the coefficients by minimizing the squared residual function. The linear equation assigns one scale factor represented by a coefficient to each covariate and an additional coefficient called the intercept or the bias coefficient which gives the line an additional degree of freedom allowing to move up and down a two-dimensional plot.

1. Missing data for numeric variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 1
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 1
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 5
2. Missing data for categorical variables were imputed using the most frequent value.
    * <span style="color: #FF0000">HDICAP</span>: Null.Count = 1


```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cancer_rate_cleaned.columns),
                                                  list(cancer_rate_cleaned.dtypes),
                                                  list([len(cancer_rate_cleaned)] * len(cancer_rate_cleaned.columns)),
                                                  list(cancer_rate_cleaned.count()),
                                                  list(cancer_rate_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>158</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='object')
```


```python
##################################
# Formulating the cleaned dataset
# with numeric columns only
##################################
cancer_rate_cleaned_numeric = cancer_rate_cleaned.select_dtypes(include='number')
```


```python
##################################
# Taking a snapshot of the cleaned dataset
##################################
cancer_rate_cleaned_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>452.4</td>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>46.252480</td>
      <td>5.719031e+05</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>51722.06900</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>422.9</td>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>38.562911</td>
      <td>8.015803e+04</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>NaN</td>
      <td>19.331586</td>
      <td>41760.59478</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>372.8</td>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>65.495718</td>
      <td>5.949773e+04</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>85420.19086</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>362.2</td>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>44.363367</td>
      <td>5.505181e+06</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>63528.63430</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>351.1</td>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>65.499675</td>
      <td>4.113555e+04</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>60915.42440</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cancer_rate_imputed_numeric_array = iterative_imputer.fit_transform(cancer_rate_cleaned_numeric)
```


```python
##################################
# Transforming the imputed data
# from an array to a dataframe
##################################
cancer_rate_imputed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array, 
                                           columns = cancer_rate_cleaned_numeric.columns)
```


```python
##################################
# Taking a snapshot of the imputed dataset
##################################
cancer_rate_imputed_numeric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>452.4</td>
      <td>98380.63601</td>
      <td>86.241</td>
      <td>1.235701</td>
      <td>83.200000</td>
      <td>7.2</td>
      <td>4.941054</td>
      <td>46.252480</td>
      <td>5.719031e+05</td>
      <td>131484.763200</td>
      <td>17.421315</td>
      <td>14.772658</td>
      <td>24.893584</td>
      <td>3.335312</td>
      <td>51722.06900</td>
      <td>60.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>422.9</td>
      <td>77541.76438</td>
      <td>86.699</td>
      <td>2.204789</td>
      <td>82.256098</td>
      <td>7.2</td>
      <td>4.354730</td>
      <td>38.562911</td>
      <td>8.015803e+04</td>
      <td>32241.937000</td>
      <td>37.570126</td>
      <td>6.160799</td>
      <td>59.475540</td>
      <td>19.331586</td>
      <td>41760.59478</td>
      <td>56.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>372.8</td>
      <td>198405.87500</td>
      <td>63.653</td>
      <td>1.029111</td>
      <td>82.556098</td>
      <td>5.3</td>
      <td>5.684596</td>
      <td>65.495718</td>
      <td>5.949773e+04</td>
      <td>15252.824630</td>
      <td>11.351720</td>
      <td>6.768228</td>
      <td>0.274092</td>
      <td>72.367281</td>
      <td>85420.19086</td>
      <td>57.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>362.2</td>
      <td>130941.63690</td>
      <td>82.664</td>
      <td>0.964348</td>
      <td>76.980488</td>
      <td>2.3</td>
      <td>5.302060</td>
      <td>44.363367</td>
      <td>5.505181e+06</td>
      <td>748241.402900</td>
      <td>33.866926</td>
      <td>13.032828</td>
      <td>3.343170</td>
      <td>36.240985</td>
      <td>63528.63430</td>
      <td>51.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>351.1</td>
      <td>113300.60110</td>
      <td>88.116</td>
      <td>0.291641</td>
      <td>81.602439</td>
      <td>4.1</td>
      <td>6.826140</td>
      <td>65.499675</td>
      <td>4.113555e+04</td>
      <td>7778.773921</td>
      <td>15.711000</td>
      <td>4.691237</td>
      <td>56.914456</td>
      <td>145.785100</td>
      <td>60915.42440</td>
      <td>77.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='category')
```


```python
##################################
# Imputing the missing data
# for categorical columns with
# the most frequent category
##################################
cancer_rate_cleaned_categorical['HDICAT'].fillna(cancer_rate_cleaned_categorical['HDICAT'].mode()[0], inplace=True)
cancer_rate_imputed_categorical = cancer_rate_cleaned_categorical.reset_index(drop=True)
```


```python
##################################
# Formulating the imputed dataset
##################################
cancer_rate_imputed = pd.concat([cancer_rate_imputed_numeric,cancer_rate_imputed_categorical], axis=1, join='inner')  
```


```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate_imputed.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate_imputed.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate_imputed)] * len(cancer_rate_imputed.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate_imputed.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate_imputed.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
imputed_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                                  data_type_list,
                                                  row_count_list,
                                                  non_null_count_list,
                                                  null_count_list,
                                                  fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(imputed_column_quality_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed for 5 numeric variables with Outlier.Ratio>0.10 and marginal to high Skewness.
    * <span style="color: #FF0000">PM2EXP</span>: Outlier.Count = 37, Outlier.Ratio = 0.226, Skewness=-3.061
    * <span style="color: #FF0000">GHGEMI</span>: Outlier.Count = 27, Outlier.Ratio = 0.165, Skewness=+9.299
    * <span style="color: #FF0000">GDPCAP</span>: Outlier.Count = 22, Outlier.Ratio = 0.134, Skewness=+2.311
    * <span style="color: #FF0000">POPDEN</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+9.972
    * <span style="color: #FF0000">METEMI</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+5.688
2. Minimal number of outliers observed for 5 numeric variables with Outlier.Ratio<0.10 and normal Skewness.
    * <span style="color: #FF0000">TUBINC</span>: Outlier.Count = 12, Outlier.Ratio = 0.073, Skewness=+1.747
    * <span style="color: #FF0000">CO2EMI</span>: Outlier.Count = 11, Outlier.Ratio = 0.067, Skewness=+2.693
    * <span style="color: #FF0000">GDPPER</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+1.554
    * <span style="color: #FF0000">EPISCO</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+0.635
    * <span style="color: #FF0000">CANRAT</span>: Outlier.Count = 2, Outlier.Ratio = 0.012, Skewness=+0.910


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cancer_rate_imputed_numeric = cancer_rate_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = list(cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cancer_rate_imputed_numeric_q1 = cancer_rate_imputed_numeric.quantile(0.25)
cancer_rate_imputed_numeric_q3 = cancer_rate_imputed_numeric.quantile(0.75)
cancer_rate_imputed_numeric_iqr = cancer_rate_imputed_numeric_q3 - cancer_rate_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((cancer_rate_imputed_numeric < (cancer_rate_imputed_numeric_q1 - 1.5 * cancer_rate_imputed_numeric_iqr)) | (cancer_rate_imputed_numeric > (cancer_rate_imputed_numeric_q3 + 1.5 * cancer_rate_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(cancer_rate_imputed_numeric)] * len(cancer_rate_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>0.910128</td>
      <td>2</td>
      <td>163</td>
      <td>0.012270</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDPPER</td>
      <td>1.554434</td>
      <td>3</td>
      <td>163</td>
      <td>0.018405</td>
    </tr>
    <tr>
      <th>2</th>
      <td>URBPOP</td>
      <td>-0.212327</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POPGRO</td>
      <td>-0.181666</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LIFEXP</td>
      <td>-0.329704</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TUBINC</td>
      <td>1.747962</td>
      <td>12</td>
      <td>163</td>
      <td>0.073620</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DTHCMD</td>
      <td>0.930709</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AGRLND</td>
      <td>0.035315</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GHGEMI</td>
      <td>9.299960</td>
      <td>27</td>
      <td>163</td>
      <td>0.165644</td>
    </tr>
    <tr>
      <th>9</th>
      <td>METEMI</td>
      <td>5.688689</td>
      <td>20</td>
      <td>163</td>
      <td>0.122699</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FORARE</td>
      <td>0.556183</td>
      <td>0</td>
      <td>163</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CO2EMI</td>
      <td>2.693585</td>
      <td>11</td>
      <td>163</td>
      <td>0.067485</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PM2EXP</td>
      <td>-3.061617</td>
      <td>37</td>
      <td>163</td>
      <td>0.226994</td>
    </tr>
    <tr>
      <th>13</th>
      <td>POPDEN</td>
      <td>9.972806</td>
      <td>20</td>
      <td>163</td>
      <td>0.122699</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GDPCAP</td>
      <td>2.311079</td>
      <td>22</td>
      <td>163</td>
      <td>0.134969</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EPISCO</td>
      <td>0.635994</td>
      <td>3</td>
      <td>163</td>
      <td>0.018405</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cancer_rate_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_imputed_numeric, x=column)
```


    
![png](output_121_0.png)
    



    
![png](output_121_1.png)
    



    
![png](output_121_2.png)
    



    
![png](output_121_3.png)
    



    
![png](output_121_4.png)
    



    
![png](output_121_5.png)
    



    
![png](output_121_6.png)
    



    
![png](output_121_7.png)
    



    
![png](output_121_8.png)
    



    
![png](output_121_9.png)
    



    
![png](output_121_10.png)
    



    
![png](output_121_11.png)
    



    
![png](output_121_12.png)
    



    
![png](output_121_13.png)
    



    
![png](output_121_14.png)
    



    
![png](output_121_15.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

[Pearson’s Correlation Coefficient](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) is a parametric measure of the linear correlation for a pair of features by calculating the ratio between their covariance and the product of their standard deviations. The presence of high absolute correlation values indicate the univariate association between the numeric predictors and the numeric response.

1. Majority of the numeric variables reported moderate to high correlation which were statistically significant.
2. Among pairwise combinations of numeric variables, high Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">GDPPER</span> and <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient = +0.921
    * <span style="color: #FF0000">GHGEMI</span> and <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.905
3. Among the highly correlated pairs, variables with the lowest correlation against the target variable were removed.
    * <span style="color: #FF0000">GDPPER</span>: Pearson.Correlation.Coefficient = +0.690
    * <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.062
4. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **16 columns** (variables)
        * **1/16 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/16 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **13/16 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/16 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'})  
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cancer_rate_imputed_numeric_correlation_pairs = {}
cancer_rate_imputed_numeric_columns = cancer_rate_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cancer_rate_imputed_numeric_columns, 2):
    cancer_rate_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cancer_rate_imputed_numeric.loc[:, numeric_column_a], 
        cancer_rate_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cancer_rate_imputed_numeric_summary = cancer_rate_imputed_numeric.from_dict(cancer_rate_imputed_numeric_correlation_pairs, orient='index')
cancer_rate_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cancer_rate_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GDPPER_GDPCAP</th>
      <td>0.921009</td>
      <td>8.173822e-68</td>
    </tr>
    <tr>
      <th>GHGEMI_METEMI</th>
      <td>0.905121</td>
      <td>1.087643e-61</td>
    </tr>
    <tr>
      <th>POPGRO_DTHCMD</th>
      <td>0.759470</td>
      <td>7.124695e-32</td>
    </tr>
    <tr>
      <th>GDPPER_LIFEXP</th>
      <td>0.755792</td>
      <td>2.052275e-31</td>
    </tr>
    <tr>
      <th>CANRAT_EPISCO</th>
      <td>0.712599</td>
      <td>1.445594e-26</td>
    </tr>
    <tr>
      <th>CANRAT_GDPCAP</th>
      <td>0.696991</td>
      <td>4.991271e-25</td>
    </tr>
    <tr>
      <th>GDPCAP_EPISCO</th>
      <td>0.696707</td>
      <td>5.312642e-25</td>
    </tr>
    <tr>
      <th>CANRAT_LIFEXP</th>
      <td>0.692318</td>
      <td>1.379448e-24</td>
    </tr>
    <tr>
      <th>CANRAT_GDPPER</th>
      <td>0.686787</td>
      <td>4.483016e-24</td>
    </tr>
    <tr>
      <th>LIFEXP_GDPCAP</th>
      <td>0.683834</td>
      <td>8.321371e-24</td>
    </tr>
    <tr>
      <th>GDPPER_EPISCO</th>
      <td>0.680814</td>
      <td>1.554608e-23</td>
    </tr>
    <tr>
      <th>GDPPER_URBPOP</th>
      <td>0.666399</td>
      <td>2.778872e-22</td>
    </tr>
    <tr>
      <th>GDPPER_CO2EMI</th>
      <td>0.654956</td>
      <td>2.451320e-21</td>
    </tr>
    <tr>
      <th>TUBINC_DTHCMD</th>
      <td>0.643615</td>
      <td>1.936081e-20</td>
    </tr>
    <tr>
      <th>URBPOP_LIFEXP</th>
      <td>0.623997</td>
      <td>5.669778e-19</td>
    </tr>
    <tr>
      <th>LIFEXP_EPISCO</th>
      <td>0.620271</td>
      <td>1.048393e-18</td>
    </tr>
    <tr>
      <th>URBPOP_GDPCAP</th>
      <td>0.559181</td>
      <td>8.624533e-15</td>
    </tr>
    <tr>
      <th>CO2EMI_GDPCAP</th>
      <td>0.550221</td>
      <td>2.782997e-14</td>
    </tr>
    <tr>
      <th>URBPOP_CO2EMI</th>
      <td>0.550046</td>
      <td>2.846393e-14</td>
    </tr>
    <tr>
      <th>LIFEXP_CO2EMI</th>
      <td>0.531305</td>
      <td>2.951829e-13</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cancer_rate_imputed_numeric_correlation = cancer_rate_imputed_numeric.corr()
mask = np.triu(cancer_rate_imputed_numeric_correlation)
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_126_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cancer_rate_imputed_numeric_correlation_p_values = correlation_significance(cancer_rate_imputed_numeric)                     
mask = np.invert(np.tril(cancer_rate_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)  
```


    
![png](output_128_0.png)
    



```python
##################################
# Filtering out one among the 
# highly correlated variable pairs with
# lesser Pearson.Correlation.Coefficient
# when compared to the target variable
##################################
cancer_rate_imputed_numeric.drop(['GDPPER','METEMI'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_imputed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 14)


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

[Yeo-Johnson Transformation](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) applies a new family of distributions that can be used without restrictions, extending many of the good properties of the Box-Cox power family. Similar to the Box-Cox transformation, the method also estimates the optimal value of lambda but has the ability to transform both positive and negative values by inflating low variance data and deflating high variance data to create a more uniform data set. While there are no restrictions in terms of the applicable values, the interpretability of the transformed values is more diminished as compared to the other methods.

1. A Yeo-Johnson transformation was applied to all numeric variables to improve distributional shape.
2. Most variables achieved symmetrical distributions with minimal outliers after transformation.
3. One variable which remained skewed even after applying shape transformation was removed.
    * <span style="color: #FF0000">PM2EXP</span> 
4. The transformed dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cancer_rate_imputed_numeric_array = yeo_johnson_transformer.fit_transform(cancer_rate_imputed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cancer_rate_transformed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array,
                                               columns=cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_transformed_numeric, x=column)
```


    
![png](output_134_0.png)
    



    
![png](output_134_1.png)
    



    
![png](output_134_2.png)
    



    
![png](output_134_3.png)
    



    
![png](output_134_4.png)
    



    
![png](output_134_5.png)
    



    
![png](output_134_6.png)
    



    
![png](output_134_7.png)
    



    
![png](output_134_8.png)
    



    
![png](output_134_9.png)
    



    
![png](output_134_10.png)
    



    
![png](output_134_11.png)
    



    
![png](output_134_12.png)
    



    
![png](output_134_13.png)
    



```python
##################################
# Filtering out the column
# which remained skewed even
# after applying shape transformation
##################################
cancer_rate_transformed_numeric.drop(['PM2EXP'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_transformed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 13)


### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

1. All numeric variables were transformed using the standardization method to achieve a comparable scale between values.
4. The scaled dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting standardization
# to transform the values of the 
# variables into comparable scale
##################################
standardization_scaler = StandardScaler()
cancer_rate_transformed_numeric_array = standardization_scaler.fit_transform(cancer_rate_transformed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the scaled data
##################################
cancer_rate_scaled_numeric = pd.DataFrame(cancer_rate_transformed_numeric_array,
                                          columns=cancer_rate_transformed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_scaled_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_scaled_numeric, x=column)
```


    
![png](output_140_0.png)
    



    
![png](output_140_1.png)
    



    
![png](output_140_2.png)
    



    
![png](output_140_3.png)
    



    
![png](output_140_4.png)
    



    
![png](output_140_5.png)
    



    
![png](output_140_6.png)
    



    
![png](output_140_7.png)
    



    
![png](output_140_8.png)
    



    
![png](output_140_9.png)
    



    
![png](output_140_10.png)
    



    
![png](output_140_11.png)
    



    
![png](output_140_12.png)
    


### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

1. One-hot encoding was applied to the <span style="color: #FF0000">HDICAP_VH</span> variable resulting to 4 additional columns in the dataset:
    * <span style="color: #FF0000">HDICAP_L</span>
    * <span style="color: #FF0000">HDICAP_M</span>
    * <span style="color: #FF0000">HDICAP_H</span>
    * <span style="color: #FF0000">HDICAP_VH</span>


```python
##################################
# Formulating the categorical column
# for encoding transformation
##################################
cancer_rate_categorical_encoded = pd.DataFrame(cancer_rate_cleaned_categorical.loc[:, 'HDICAT'].to_list(),
                                               columns=['HDICAT'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the categorical column
##################################
cancer_rate_categorical_encoded = pd.get_dummies(cancer_rate_categorical_encoded, columns=['HDICAT'])
```

### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

1. The preprocessed dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/18 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **4/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT_L</span>
             * <span style="color: #FF0000">HDICAT_M</span>
             * <span style="color: #FF0000">HDICAT_H</span>
             * <span style="color: #FF0000">HDICAT_VH</span>


```python
##################################
# Consolidating both numeric columns
# and encoded categorical columns
##################################
cancer_rate_preprocessed = pd.concat([cancer_rate_scaled_numeric,cancer_rate_categorical_encoded], axis=1, join='inner')  
```


```python
##################################
# Performing a general exploration of the consolidated dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_preprocessed.shape)
```

    Dataset Dimensions: 
    


    (163, 17)


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Bivariate analysis identified individual predictors with generally linear relationship to the target variable based on visual inspection.
2. Increasing values for the following predictors correspond to higher <span style="color: #FF0000">CANRAT</span> measurements: 
    * <span style="color: #FF0000">URBPOP</span>
    * <span style="color: #FF0000">LIFEXP</span>    
    * <span style="color: #FF0000">CO2EMI</span>    
    * <span style="color: #FF0000">GDPCAP</span>    
    * <span style="color: #FF0000">EPISCO</span>    
    * <span style="color: #FF0000">HDICAP_VH</span>
3. Decreasing values for the following predictors correspond to higher <span style="color: #FF0000">CANRAT</span> measurements: 
    * <span style="color: #FF0000">POPGRO</span>
    * <span style="color: #FF0000">TUBINC</span>    
    * <span style="color: #FF0000">DTHCMD</span> 
    * <span style="color: #FF0000">HDICAP_L</span>
    * <span style="color: #FF0000">HDICAP_M</span>
4. Values for the following predictors did not affect <span style="color: #FF0000">CANRAT</span> measurements: 
    * <span style="color: #FF0000">AGRLND</span>
    * <span style="color: #FF0000">GHGEMI</span>    
    * <span style="color: #FF0000">FORARE</span> 
    * <span style="color: #FF0000">POPDEN</span> 
    * <span style="color: #FF0000">HDICAP_H</span>


```python
##################################
# Segregating the target
# and predictor variable lists
##################################
cancer_rate_preprocessed_target = ['CANRAT']
cancer_rate_preprocessed_predictors = cancer_rate_preprocessed.drop('CANRAT', axis=1).columns
```


```python
##################################
# Segregating the target
# and predictor variable names
##################################
y_variable = 'CANRAT'
x_variables = cancer_rate_preprocessed_predictors
```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 8
num_cols = 2
```


```python
##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual scatterplots
# for all scaled numeric columns
##################################
for i, x_variable in enumerate(x_variables):
    ax = axes[i]
    ax.scatter(cancer_rate_preprocessed[x_variable],cancer_rate_preprocessed[y_variable])
    ax.set_title(f'{y_variable} Versus {x_variable}')
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_152_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Pearson correlation coefficient is equal to zero 
    * **Alternative**: Pearson correlation coefficient is not equal to zero    
2. There is sufficient evidence to conclude of a statistically significant linear relationship between the <span style="color: #FF0000">CANRAT</span> target variable and 10 of the 12 numeric predictors given their high Pearson correlation coefficient values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient=+0.735, Correlation.PValue=0.000
    * <span style="color: #FF0000">LIFEXP</span>: Pearson.Correlation.Coefficient=+0.702, Correlation.PValue=0.000   
    * <span style="color: #FF0000">DTHCMD</span>: Pearson.Correlation.Coefficient=-0.687, Correlation.PValue=0.000 
    * <span style="color: #FF0000">EPISCO</span>: Pearson.Correlation.Coefficient=+0.648, Correlation.PValue=0.000 
    * <span style="color: #FF0000">TUBINC</span>: Pearson.Correlation.Coefficient=+0.628, Correlation.PValue=0.000 
    * <span style="color: #FF0000">CO2EMI</span>: Pearson.Correlation.Coefficient=+0.585, Correlation.PValue=0.000  
    * <span style="color: #FF0000">POPGRO</span>: Pearson.Correlation.Coefficient=-0.498, Correlation.PValue=0.000
    * <span style="color: #FF0000">URBPOP</span>: Pearson.Correlation.Coefficient=+0.479, Correlation.PValue=0.000   
    * <span style="color: #FF0000">GHGEMI</span>: Pearson.Correlation.Coefficient=+0.232, Correlation.PValue=0.002
    * <span style="color: #FF0000">FORARE</span>: Pearson.Correlation.Coefficient=+0.165, Correlation.PValue=0.035
3. The relationship between the categorical predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups 0 and 1 is equal to zero 
    * **Alternative**: Difference in the means between groups 0 and 1 is not equal to zero    
2. There is sufficient evidence to conclude of a statistically significant difference between the means of <span style="color: #FF0000">CANRAT</span> measurements obtained from groups 0 and 1 in 3 of the 4 categorical predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">HDICAT_VH</span>: T.Test.Statistic=-10.605, T.Test.PValue=0.000
    * <span style="color: #FF0000">HDICAT_L</span>: T.Test.Statistic=+6.559, T.Test.PValue=0.000   
    * <span style="color: #FF0000">HDICAT_M</span>: T.Test.Statistic=+5.104, T.Test.PValue=0.000 


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_correlation_target = {}
cancer_rate_preprocessed_numeric = cancer_rate_preprocessed.drop(['HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH'], axis=1)
cancer_rate_preprocessed_numeric_columns = cancer_rate_preprocessed_numeric.columns.tolist()
for numeric_column in cancer_rate_preprocessed_numeric_columns:
    cancer_rate_preprocessed_numeric_correlation_target['CANRAT_' + numeric_column] = stats.pearsonr(
        cancer_rate_preprocessed_numeric.loc[:, 'CANRAT'], 
        cancer_rate_preprocessed_numeric.loc[:, numeric_column])
```


```python
##################################
# Formulating the pairwise correlation summary
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_summary = cancer_rate_preprocessed_numeric.from_dict(cancer_rate_preprocessed_numeric_correlation_target, orient='index')
cancer_rate_preprocessed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cancer_rate_preprocessed_numeric_summary.sort_values(by=['Correlation.PValue'], ascending=True).head(13))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_CANRAT</th>
      <td>1.000000</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>CANRAT_GDPCAP</th>
      <td>0.735131</td>
      <td>5.617239e-29</td>
    </tr>
    <tr>
      <th>CANRAT_LIFEXP</th>
      <td>0.702430</td>
      <td>1.491302e-25</td>
    </tr>
    <tr>
      <th>CANRAT_DTHCMD</th>
      <td>-0.687136</td>
      <td>4.164564e-24</td>
    </tr>
    <tr>
      <th>CANRAT_EPISCO</th>
      <td>0.648431</td>
      <td>8.136735e-21</td>
    </tr>
    <tr>
      <th>CANRAT_TUBINC</th>
      <td>-0.628877</td>
      <td>2.503346e-19</td>
    </tr>
    <tr>
      <th>CANRAT_CO2EMI</th>
      <td>0.585452</td>
      <td>2.251585e-16</td>
    </tr>
    <tr>
      <th>CANRAT_POPGRO</th>
      <td>-0.498457</td>
      <td>1.278437e-11</td>
    </tr>
    <tr>
      <th>CANRAT_URBPOP</th>
      <td>0.479386</td>
      <td>9.543704e-11</td>
    </tr>
    <tr>
      <th>CANRAT_GHGEMI</th>
      <td>0.232488</td>
      <td>2.822914e-03</td>
    </tr>
    <tr>
      <th>CANRAT_FORARE</th>
      <td>0.165265</td>
      <td>3.500993e-02</td>
    </tr>
    <tr>
      <th>CANRAT_AGRLND</th>
      <td>-0.024520</td>
      <td>7.560347e-01</td>
    </tr>
    <tr>
      <th>CANRAT_POPDEN</th>
      <td>0.001902</td>
      <td>9.807807e-01</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the t-test 
# statistic and p-values
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_ttest_target = {}
cancer_rate_preprocessed_categorical = cancer_rate_preprocessed[['CANRAT','HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH']]
cancer_rate_preprocessed_categorical_columns = ['HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH']
for categorical_column in cancer_rate_preprocessed_categorical_columns:
    group_0 = cancer_rate_preprocessed_categorical[cancer_rate_preprocessed_categorical.loc[:,categorical_column]==0]
    group_1 = cancer_rate_preprocessed_categorical[cancer_rate_preprocessed_categorical.loc[:,categorical_column]==1]
    cancer_rate_preprocessed_categorical_ttest_target['CANRAT_' + categorical_column] = stats.ttest_ind(
        group_0['CANRAT'], 
        group_1['CANRAT'], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_summary = cancer_rate_preprocessed_categorical.from_dict(cancer_rate_preprocessed_categorical_ttest_target, orient='index')
cancer_rate_preprocessed_categorical_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(cancer_rate_preprocessed_categorical_summary.sort_values(by=['T.Test.PValue'], ascending=True).head(4))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_HDICAT_VH</th>
      <td>-10.605706</td>
      <td>2.909971e-20</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_L</th>
      <td>6.559780</td>
      <td>7.003957e-10</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_M</th>
      <td>5.104986</td>
      <td>9.237518e-07</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_H</th>
      <td>-0.635957</td>
      <td>5.257075e-01</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Linear Regression Model Coefficient Estimation <a class="anchor" id="1.6"></a>

### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>

1. Among the predictor variables determined to have a statistically significant linear relationship between the <span style="color: #FF0000">CANRAT</span> target variable, only 2 were retained with absolute Pearson correlation coefficient values greater than 0.70. 
    * <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient=+0.735, Correlation.PValue=0.000
    * <span style="color: #FF0000">LIFEXP</span>: Pearson.Correlation.Coefficient=+0.702, Correlation.PValue=0.000   



```python
##################################
# Consolidating relevant numeric columns
# and encoded categorical columns
# after hypothesis testing
##################################
cancer_rate_premodelling = cancer_rate_preprocessed.drop(['DTHCMD','EPISCO','TUBINC','CO2EMI','AGRLND','POPDEN','GHGEMI','FORARE','POPGRO','URBPOP','HDICAT_VH','HDICAT_H','HDICAT_M','HDICAT_L'], axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_premodelling.shape)
```

    Dataset Dimensions: 
    


    (163, 3)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate_premodelling.dtypes)
```

    Column Names and Data Types:
    


    CANRAT    float64
    LIFEXP    float64
    GDPCAP    float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate_premodelling.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CANRAT</th>
      <th>LIFEXP</th>
      <th>GDPCAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.076468</td>
      <td>1.643195</td>
      <td>1.549766</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.962991</td>
      <td>1.487969</td>
      <td>1.407752</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.742760</td>
      <td>1.537044</td>
      <td>1.879374</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.690866</td>
      <td>0.664178</td>
      <td>1.685426</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.634224</td>
      <td>1.381877</td>
      <td>1.657777</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Gathering the pairplot for all variables
##################################
sns.pairplot(cancer_rate_premodelling, kind='reg')
plt.show()
```


    
![png](output_164_0.png)
    



```python
##################################
# Separating the target 
# and predictor columns
##################################
X = cancer_rate_premodelling.drop('CANRAT', axis = 1)
y = cancer_rate_premodelling.CANRAT
```


```python
##################################
# Formulating the train and test data
# using a 70-30 ratio
##################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 88888888)
```


```python
##################################
# Performing a general exploration of the train dataset
##################################
print('Dataset Dimensions: ')
display(X_train.shape)
```

    Dataset Dimensions: 
    


    (114, 2)


### 1.6.2 Normal Equations <a class="anchor" id="1.6.2"></a>

[Normal Equations](https://www.semanticscholar.org/paper/Nouvelles-m%C3%A9thodes-pour-la-d%C3%A9termination-des-des-Legendre/21c9090e226ab449ffb608ddb2cb925911a61f24) are a system of equations whose solution is the Ordinary Least Squares (OLS) estimator of the regression coefficients and which are derived from the first-order condition of the least squares minimization problem. These equations are obtained by setting equal to zero the partial derivatives of the sum of squared errors (least squares). This approach is a closed-form solution and a one-step algorithm used to analytically find the coefficients that minimize the loss function.

1. Applying normal equations, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = -0.00061
    * <span style="color: #FF0000">LIFEXP</span> = +0.23342
    * <span style="color: #FF0000">GDPCAP</span> = +0.50808
2. The [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API was implemented which generated the same regression coefficient estimates.
3. These estimated coefficients will be the baseline values from which all gradient descent algorithm-derived coefficients will be compared with.



```python
##################################
# Defining the components
# for matrix algebra computations
# using Normal Equations
##################################
num_observations = X_train.shape[0]
constant_array = np.ones(num_observations)
x_train_matrix = np.array([constant_array,X_train.LIFEXP,X_train.GDPCAP]).T
```


```python
##################################
# Consolidating the regression coefficients
# obtained using the Normal Equations
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_estimates = pd.DataFrame(np.linalg.inv(np.dot(x_train_matrix.T,x_train_matrix)).dot(x_train_matrix.T).dot(y_train))
linear_regression_normal_equations = pd.concat([linear_regression_coefficients, linear_regression_estimates], axis=1)
linear_regression_normal_equations.columns = ['Coefficient', 'Estimate']
linear_regression_normal_equations.reset_index(inplace=True, drop=True)
display(linear_regression_normal_equations)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>-0.000606</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.233418</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.508083</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Defining the linear regression model
# using the Scikit-Learn package
##################################
linear_regression = LinearRegression()

##################################
# Fitting a linear regression model
##################################
linear_regression.fit(X_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




```python
##################################
# Consolidating the regression coefficients
# obtained using the Scikit-Learn package
##################################
linear_regression_intercept = pd.DataFrame(zip(["INTERCEPT"], [linear_regression.intercept_]))
linear_regression_predictors = pd.DataFrame(zip(X_train.columns, linear_regression.coef_))
linear_regression_scikitlearn_estimates = pd.concat([linear_regression_intercept, linear_regression_predictors], axis=0)
linear_regression_scikitlearn_estimates.reset_index(inplace=True, drop=True)
linear_regression_method = pd.DataFrame(["Normal_Equations"]*3)
linear_regression_scikitlearn_computations = pd.concat([linear_regression_scikitlearn_estimates,linear_regression_method], axis=1)
linear_regression_scikitlearn_computations.columns = ['Coefficient', 'Estimate','Method']
linear_regression_scikitlearn_computations.reset_index(inplace=True, drop=True)
display(linear_regression_scikitlearn_computations)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>-0.000606</td>
      <td>Normal_Equations</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.233418</td>
      <td>Normal_Equations</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.508083</td>
      <td>Normal_Equations</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Setting the regression coefficients
# determined using normal equations
# as the target coefficient estimates
##################################
intercept = -0.000606
theta_1 = 0.233418
theta_2 = 0.508083
```

### 1.6.3 Gradient Descent Algorithm with Very High Learning Rate and Low Epoch Count <a class="anchor" id="1.6.3"></a>

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. A very high learning rate (also referred to as step size or the alpha) and low epoch count were applied resulting in larger steps with lesser risks of overshooting the minimum due to a lower number of iterations.

1. The gradient descent algorithm was implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 1.00 (Very High)
    * <span style="color: #FF0000">Epochs</span> = 30 (Low)
2. The final squared loss estimate determined as 53.54705 at the 30th epoch was not optimally low as compared to those obtained using the other parameter settings.
3. Applying the gradient descent algorithm with a very high learning rate and low epoch count, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = +0.00801 (Baseline = -0.00061)
    * <span style="color: #FF0000">LIFEXP</span> = +0.28176 (Baseline = +0.23342)
    * <span style="color: #FF0000">GDPCAP</span> = +0.54996 (Baseline = +0.50808)
4. The estimated coefficients using the gradient descent algorithm with a very high learning rate and low epoch count were not fully optimized and comparable with the baseline coefficients using normal equations.



```python
##################################
# Formulating a function
# for computing the regression coefficients
# using gradient descent
##################################
def gradient_descent(learning_rate, num_iterations, theta_initial):
    # Setting the initialization values
    # Initializing the coefficient estimates
    theta = theta_initial
    # Initializing the gradient descent trajectory path
    theta_path = np.zeros((num_iterations+1,3))
    theta_path[0,:]= theta_initial
    # Initializing the loss function values
    loss_vector = np.zeros(num_iterations)
    ## Implementing the main Gradient Descent loop givne a fixed number of iterations
    for i in range(num_iterations):
        # Generating predictions
        y_predicted = np.dot(theta.T,x_train_matrix.T)
        # Computing for the loss function values
        loss_vector[i] = np.sum((y_train-y_predicted)**2)
        # Summing up the gradients across all observations and divide by the number of observations
        gradient_vector = (y_train-y_predicted).dot(x_train_matrix)/num_observations  
        # Updating the gradients
        gradient_vector = gradient_vector
        # Updating the coefficient estimates
        theta = theta + learning_rate*gradient_vector
        # Updating the trajectory of the gradient descent process
        theta_path[i+1,:]=theta
    return theta_path, loss_vector

##################################
# Formulating a function
# for plotting the gradient descent trajectory
# with respect to the initialized
# and target coefficient estimates
##################################
target_coefficient = [intercept, theta_1, theta_2]
target_coefficient_name = ["INTERCEPT","LIFEXP","GDPCAP"]
def plot_ij(theta_path, i, j, ax):
    ax.plot(target_coefficient[i], target_coefficient[j],
            marker='p', markersize=15, label='Target Coefficient', 
            color='#FF0000')
    ax.plot(theta_path[:, i],theta_path[:, j],
            color='k', linestyle='--', marker='^', 
            markersize=5, markevery=1)
    ax.plot(theta_path[0, i], theta_path[0, j], marker='d', 
            markersize=10, label='Gradient Descent Path Start', color='#00FF0088')
    ax.plot(theta_path[-1, i], theta_path[-1, j], marker='o', 
            markersize=10, label='Gradient Descent Path End', color='#0000FF88')
    ax.set(
        xlabel=target_coefficient_name[i],
        ylabel=target_coefficient_name[j])
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc='upper left')
    
##################################
# Consolidating all 
# gradient descent trajectory plots
# by pairwise combinations
# of regression coefficients
##################################
def plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial, gdtype='Gradient Descent'):
    fig = plt.figure(figsize=(12, 12))
    title = '{gdtype}: Learning Rate = {lr} | Epoch Count = {iters}'
    title = title.format(gdtype=gdtype, lr=learning_rate, 
                         iters=num_iterations, initial=theta_initial)
    fig.suptitle(title, fontsize=15)
    ax = fig.add_subplot(2, 2, 1)
    plot_ij(theta_path, 0, 1, ax)
    ax = fig.add_subplot(2, 2, 2)
    plot_ij(theta_path, 0, 2, ax)
    ax = fig.add_subplot(2, 2, 3)
    plot_ij(theta_path, 1, 2, ax)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(loss_vector)
    ax.set_ylim([0, 5000])
    ax.set_xlim([0, 300])
    ax.set(xlabel='Iterations', ylabel='Squared Loss')
    ax.grid(True)
```


```python
##################################
# Setting the initial parameters
##################################
theta_initial = np.array([3,3,3])
##################################
# Using a very high learning rate
##################################
learning_rate = 1.00
##################################
# Using a low epoch count
##################################
num_iterations = 30

##################################
# Implementing the gradient descent process
# for determining the regression coefficients
##################################
theta_path, loss_vector = gradient_descent(learning_rate, num_iterations, theta_initial)

##################################
# Consolidating the gradient descent
# trajectory plots in a 3D parameter space
# for each pair of regression coefficients
##################################
plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial)
```


    
![png](output_176_0.png)
    



```python
loss_vector_vhlearningrate_lepochcount = loss_vector[-1]
loss_vector_vhlearningrate_lepochcount
```




    53.54705756819414




```python
##################################
# Consolidating the regression coefficients
# obtained using the Gradient Descent process
# with very high learning rate
# and low epoch count
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_vhlearningrate_lepochcount = pd.DataFrame(theta_path[-1])
linear_regression_vhlearningrate_lepochcount_method = pd.DataFrame(["VeryHighLearningRate_LowEpochCount"]*3)
linear_regression_gradientdescent_vhlearningrate_lepochcount = pd.concat([linear_regression_coefficients, 
                                                                          linear_regression_vhlearningrate_lepochcount,
                                                                          linear_regression_vhlearningrate_lepochcount_method], axis=1)
linear_regression_gradientdescent_vhlearningrate_lepochcount.columns = ['Coefficient', 'Estimate', 'Method']
linear_regression_gradientdescent_vhlearningrate_lepochcount.reset_index(inplace=True, drop=True)
display(linear_regression_gradientdescent_vhlearningrate_lepochcount)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>0.008008</td>
      <td>VeryHighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.281761</td>
      <td>VeryHighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.549961</td>
      <td>VeryHighLearningRate_LowEpochCount</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.4 Gradient Descent Algorithm with Very High Learning Rate and High Epoch Count <a class="anchor" id="1.6.4"></a>

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. A very high learning rate (also referred to as step size or the alpha) and high epoch count were applied resulting in larger steps with more risks of overshooting the minimum due to a higher number of iterations.

1. The gradient descent algorithm was implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 1.00 (Very High)
    * <span style="color: #FF0000">Epochs</span> = 300 (High)
2. The final squared loss estimate determined as 52.38026 at the 300th epoch was optimally low as compared to those obtained using the other parameter settings.
3. Applying the gradient descent algorithm with a very high learning rate and high epoch count, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = -0.00061 (Baseline = -0.00061)
    * <span style="color: #FF0000">LIFEXP</span> = +0.23342 (Baseline = +0.23342)
    * <span style="color: #FF0000">GDPCAP</span> = +0.50808 (Baseline = +0.50808)
4. The estimated coefficients using the gradient descent algorithm with a very high learning rate and high epoch count were fully optimized and comparable with the baseline coefficients using normal equations.



```python
##################################
# Setting the initial parameters
##################################
theta_initial = np.array([3,3,3])
##################################
# Using a very high learning rate
##################################
learning_rate = 1.00
##################################
# Using a high epoch count
##################################
num_iterations = 300

##################################
# Implementing the gradient descent process
# for determining the regression coefficients
##################################
theta_path, loss_vector = gradient_descent(learning_rate, num_iterations, theta_initial)

##################################
# Consolidating the gradient descent
# trajectory plots in a 3D parameter space
# for each pair of regression coefficients
##################################
plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial)
```


    
![png](output_180_0.png)
    



```python
loss_vector_vhlearningrate_hepochcount = loss_vector[-1]
loss_vector_vhlearningrate_hepochcount
```




    52.38026912784726




```python
##################################
# Consolidating the regression coefficients
# obtained using the Gradient Descent process
# with very high learning rate
# and high epoch count
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_vhlearningrate_hepochcount = pd.DataFrame(theta_path[-1])
linear_regression_vhlearningrate_hepochcount_method = pd.DataFrame(["VeryHighLearningRate_HighEpochCount"]*3)
linear_regression_gradientdescent_vhlearningrate_hepochcount = pd.concat([linear_regression_coefficients, 
                                                                          linear_regression_vhlearningrate_hepochcount,
                                                                          linear_regression_vhlearningrate_hepochcount_method], axis=1)
linear_regression_gradientdescent_vhlearningrate_hepochcount.columns = ['Coefficient', 'Estimate', 'Method']
linear_regression_gradientdescent_vhlearningrate_hepochcount.reset_index(inplace=True, drop=True)
display(linear_regression_gradientdescent_vhlearningrate_hepochcount)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>-0.000606</td>
      <td>VeryHighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.233418</td>
      <td>VeryHighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.508083</td>
      <td>VeryHighLearningRate_HighEpochCount</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.5 Gradient Descent Algorithm with High Learning Rate and Low Epoch Count <a class="anchor" id="1.6.5"></a>

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. A sufficiently high learning rate (also referred to as step size or the alpha) and low epoch count were applied resulting in average steps with more risks of not reaching the minimum due to a lower number of iterations.

1. The gradient descent algorithm was implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.10 (High)
    * <span style="color: #FF0000">Epochs</span> = 30 (Low)
2. The final squared loss estimate determined as 54.31567 at the 30th epoch was not optimally low as compared to those obtained using the other parameter settings.
3. Applying the gradient descent algorithm with a high learning rate and low epoch count, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = +0.11066 (Baseline = -0.00061)
    * <span style="color: #FF0000">LIFEXP</span> = +0.31784 (Baseline = +0.23342)
    * <span style="color: #FF0000">GDPCAP</span> = +0.41305 (Baseline = +0.50808)
4. The estimated coefficients using the gradient descent algorithm with a high learning rate and low epoch count were not fully optimized and comparable with the baseline coefficients using normal equations.



```python
##################################
# Setting the initial parameters
##################################
theta_initial = np.array([3,3,3])
##################################
# Using a high learning rate
##################################
learning_rate = 0.1
##################################
# Using a low epoch count
##################################
num_iterations = 30

##################################
# Implementing the gradient descent process
# for determining the regression coefficients
##################################
theta_path, loss_vector = gradient_descent(learning_rate, num_iterations, theta_initial)

##################################
# Consolidating the gradient descent
# trajectory plots in a 3D parameter space
# for each pair of regression coefficients
##################################
plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial)
```


    
![png](output_184_0.png)
    



```python
loss_vector_hlearningrate_lepochcount = loss_vector[-1]
loss_vector_hlearningrate_lepochcount
```




    54.315668691393675




```python
##################################
# Consolidating the regression coefficients
# obtained using the Gradient Descent process
# with high learning rate
# and low epoch count
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_hlearningrate_lepochcount = pd.DataFrame(theta_path[-1])
linear_regression_hlearningrate_lepochcount_method = pd.DataFrame(["HighLearningRate_LowEpochCount"]*3)
linear_regression_gradientdescent_hlearningrate_lepochcount = pd.concat([linear_regression_coefficients, 
                                                                         linear_regression_hlearningrate_lepochcount,
                                                                         linear_regression_hlearningrate_lepochcount_method], axis=1)
linear_regression_gradientdescent_hlearningrate_lepochcount.columns = ['Coefficient', 'Estimate', 'Method']
linear_regression_gradientdescent_hlearningrate_lepochcount.reset_index(inplace=True, drop=True)
display(linear_regression_gradientdescent_hlearningrate_lepochcount)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>0.110662</td>
      <td>HighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.317843</td>
      <td>HighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.413052</td>
      <td>HighLearningRate_LowEpochCount</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.6 Gradient Descent Algorithm with High Learning Rate and High Epoch Count <a class="anchor" id="1.6.6"></a>

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. A sufficiently high learning rate (also referred to as step size or the alpha) and low epoch count were applied resulting in average steps with lesser risks of not reaching the minimum as compensated by the higher number of iterations.

1. The gradient descent algorithm was implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.10 (High)
    * <span style="color: #FF0000">Epochs</span> = 300 (High)
2. The final squared loss estimate determined as 52.38063 at the 300th epoch was not optimally low as compared to those obtained using the other parameter settings.
3. Applying the gradient descent algorithm with a high learning rate and high epoch count, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = -0.00058 (Baseline = -0.00061)
    * <span style="color: #FF0000">LIFEXP</span> = +0.23703 (Baseline = +0.23342)
    * <span style="color: #FF0000">GDPCAP</span> = +0.50442 (Baseline = +0.50808)
4. The estimated coefficients using the gradient descent algorithm with a high learning rate and high epoch count, while not fully optimized, were sufficiently comparable with the baseline coefficients using normal equations.



```python
##################################
# Setting the initial parameters
##################################
theta_initial = np.array([3,3,3])
##################################
# Using a high learning rate
##################################
learning_rate = 0.1
##################################
# Using a high epoch count
##################################
num_iterations = 300

##################################
# Implementing the gradient descent process
# for determining the regression coefficients
##################################
theta_path, loss_vector = gradient_descent(learning_rate, num_iterations, theta_initial)

##################################
# Consolidating the gradient descent
# trajectory plots in a 3D parameter space
# for each pair of regression coefficients
##################################
plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial)
```


    
![png](output_188_0.png)
    



```python
loss_vector_hlearningrate_hepochcount = loss_vector[-1]
loss_vector_hlearningrate_hepochcount
```




    52.380634424224674




```python
##################################
# Consolidating the regression coefficients
# obtained using the Gradient Descent process
# with high learning rate
# and high epoch count
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_hlearningrate_hepochcount = pd.DataFrame(theta_path[-1])
linear_regression_hlearningrate_hepochcount_method = pd.DataFrame(["HighLearningRate_HighEpochCount"]*3)
linear_regression_gradientdescent_hlearningrate_hepochcount = pd.concat([linear_regression_coefficients, 
                                                                         linear_regression_hlearningrate_hepochcount,
                                                                         linear_regression_hlearningrate_hepochcount_method], axis=1)
linear_regression_gradientdescent_hlearningrate_hepochcount.columns = ['Coefficient', 'Estimate', 'Method']
linear_regression_gradientdescent_hlearningrate_hepochcount.reset_index(inplace=True, drop=True)
display(linear_regression_gradientdescent_hlearningrate_hepochcount)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>-0.000585</td>
      <td>HighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.237033</td>
      <td>HighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.504420</td>
      <td>HighLearningRate_HighEpochCount</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.7 Gradient Descent Algorithm with Low Learning Rate and Low Epoch Count <a class="anchor" id="1.6.7"></a>

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. A low learning rate (also referred to as step size or the alpha) and low epoch count were applied resulting in smaller steps with higher risks of not reaching the minimum due to the smaller number of iterations.

1. The gradient descent algorithm was implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01 (Low)
    * <span style="color: #FF0000">Epochs</span> = 30 (Low)
2. The final squared loss estimate determined as 1631.70371 at the 30th epoch was not optimally low as compared to those obtained using the other parameter settings.
3. Applying the gradient descent algorithm with a low learning rate and low epoch count, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = +2.13385 (Baseline = -0.00061)
    * <span style="color: #FF0000">LIFEXP</span> = +1.81162 (Baseline = +0.23342)
    * <span style="color: #FF0000">GDPCAP</span> = +1.83111 (Baseline = +0.50808)
4. The estimated coefficients using the gradient descent algorithm with a low learning rate and low epoch count were not fully optimized and comparable with the baseline coefficients using normal equations.



```python
##################################
# Setting the initial parameters
##################################
theta_initial = np.array([3,3,3])
##################################
# Using a low learning rate
##################################
learning_rate = 0.01
##################################
# Using a low epoch count
##################################
num_iterations = 30

##################################
# Implementing the gradient descent process
# for determining the regression coefficients
##################################
theta_path, loss_vector = gradient_descent(learning_rate, num_iterations, theta_initial)

##################################
# Consolidating the gradient descent
# trajectory plots in a 3D parameter space
# for each pair of regression coefficients
##################################
plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial)
```


    
![png](output_192_0.png)
    



```python
loss_vector_llearningrate_lepochcount = loss_vector[-1]
loss_vector_llearningrate_lepochcount
```




    1631.7037156372799




```python
##################################
# Consolidating the regression coefficients
# obtained using the Gradient Descent process
# with low learning rate
# and low epoch count
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_llearningrate_lepochcount = pd.DataFrame(theta_path[-1])
linear_regression_llearningrate_lepochcount_method = pd.DataFrame(["LowLearningRate_LowEpochCount"]*3)
linear_regression_gradientdescent_llearningrate_lepochcount = pd.concat([linear_regression_coefficients, 
                                                                         linear_regression_llearningrate_lepochcount,
                                                                         linear_regression_llearningrate_lepochcount_method], axis=1)
linear_regression_gradientdescent_llearningrate_lepochcount.columns = ['Coefficient', 'Estimate', 'Method']
linear_regression_gradientdescent_llearningrate_lepochcount.reset_index(inplace=True, drop=True)
display(linear_regression_gradientdescent_llearningrate_lepochcount)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>2.133856</td>
      <td>LowLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>1.811625</td>
      <td>LowLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>1.831115</td>
      <td>LowLearningRate_LowEpochCount</td>
    </tr>
  </tbody>
</table>
</div>


### 1.6.8 Gradient Descent Algorithm with Low Learning Rate and High Epoch Count <a class="anchor" id="1.6.8"></a>

[Gradient descent](https://link.springer.com/book/10.1007/978-1-4757-3462-1) minimizes the loss function parameterized by the model’s coefficients based on the direction and learning rate factors which determine the partial derivative calculations of future iterations, allowing the algorithm to gradually arrive at the local or global minimum considered the point of convergence. This particular implementation used Batch Gradient Descent which computes the gradient of the loss function with respect to the parameters for the entire data set. A low learning rate (also referred to as step size or the alpha) and low epoch count were applied resulting in smaller steps with lesser risks of not reaching the minimum as compensated by the smaller number of iterations.

1. The gradient descent algorithm was implemented with parameter settings described as follows:
    * <span style="color: #FF0000">Learning Rate</span> = 0.01 (Low)
    * <span style="color: #FF0000">Epochs</span> = 300 (High)
2. The final squared loss estimate determined as 54.50204 at the 300th epoch was not optimally low as compared to those obtained using the other parameter settings.
3. Applying the gradient descent algorithm with a low learning rate and high epoch count, the estimated linear regression coefficients for the given data are as follows:
    * <span style="color: #FF0000">INTERCEPT</span> = +0.12806 (Baseline = -0.00061)
    * <span style="color: #FF0000">LIFEXP</span> = +0.32058 (Baseline = +0.23342)
    * <span style="color: #FF0000">GDPCAP</span> = +0.41551 (Baseline = +0.50808)
4. The estimated coefficients using the gradient descent algorithm with a low learning rate and high epoch count were not fully optimized and comparable with the baseline coefficients using normal equations.



```python
##################################
# Setting the initial parameters
##################################
theta_initial = np.array([3,3,3])
##################################
# Using a low learning rate
##################################
learning_rate = 0.01
##################################
# Using a high epoch count
##################################
num_iterations = 300

##################################
# Implementing the gradient descent process
# for determining the regression coefficients
##################################
theta_path, loss_vector = gradient_descent(learning_rate, num_iterations, theta_initial)

##################################
# Consolidating the gradient descent
# trajectory plots in a 3D parameter space
# for each pair of regression coefficients
##################################
plot_all(theta_path, loss_vector, learning_rate, num_iterations, theta_initial)
```


    
![png](output_196_0.png)
    



```python
loss_vector_llearningrate_hepochcount = loss_vector[-1]
loss_vector_llearningrate_hepochcount
```




    54.50204042343665




```python
##################################
# Consolidating the regression coefficients
# obtained using the Gradient Descent process
# with low learning rate
# and high epoch count
##################################
linear_regression_coefficients = pd.DataFrame(["INTERCEPT","LIFEXP","GDPCAP"])
linear_regression_llearningrate_hepochcount = pd.DataFrame(theta_path[-1])
linear_regression_llearningrate_hepochcount_method = pd.DataFrame(["LowLearningRate_HighEpochCount"]*3)
linear_regression_gradientdescent_llearningrate_hepochcount = pd.concat([linear_regression_coefficients, 
                                                                         linear_regression_llearningrate_hepochcount,
                                                                         linear_regression_llearningrate_hepochcount_method], axis=1)
linear_regression_gradientdescent_llearningrate_hepochcount.columns = ['Coefficient', 'Estimate', 'Method']
linear_regression_gradientdescent_llearningrate_hepochcount.reset_index(inplace=True, drop=True)
display(linear_regression_gradientdescent_llearningrate_hepochcount)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>0.128061</td>
      <td>LowLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.320584</td>
      <td>LowLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.415517</td>
      <td>LowLearningRate_HighEpochCount</td>
    </tr>
  </tbody>
</table>
</div>


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. The gradient descent parameter setting which estimated sufficiently comparable coefficients with the baseline values are as follows:
    * <span style="color: #FF0000">VHLR_HEC</span> = Very High Learning Rate (1.0) and High Epoch Count (300)
    * <span style="color: #FF0000">HLR_HEC</span> = High Learning Rate (0.1) and High Epoch Count (300)
2. The choice of **Learning Rate** and **Epoch Count** in the implementation of the gradient descent algorithm are critical to achieving fully optimized coefficients while maintaining minimal squared loss estimates.



```python
##################################
# Consolidating the regression coefficients
# obtained using all estimation methods
##################################
linear_regression_methods = pd.concat([linear_regression_scikitlearn_computations, 
                                       linear_regression_gradientdescent_vhlearningrate_lepochcount,
                                       linear_regression_gradientdescent_vhlearningrate_hepochcount,
                                       linear_regression_gradientdescent_hlearningrate_lepochcount,
                                       linear_regression_gradientdescent_hlearningrate_hepochcount,
                                       linear_regression_gradientdescent_llearningrate_lepochcount,
                                       linear_regression_gradientdescent_llearningrate_hepochcount], axis=0)
linear_regression_methods.reset_index(inplace=True, drop=True)
display(linear_regression_methods)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
      <th>Estimate</th>
      <th>Method</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>INTERCEPT</td>
      <td>-0.000606</td>
      <td>Normal_Equations</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LIFEXP</td>
      <td>0.233418</td>
      <td>Normal_Equations</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPCAP</td>
      <td>0.508083</td>
      <td>Normal_Equations</td>
    </tr>
    <tr>
      <th>3</th>
      <td>INTERCEPT</td>
      <td>0.008008</td>
      <td>VeryHighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LIFEXP</td>
      <td>0.281761</td>
      <td>VeryHighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GDPCAP</td>
      <td>0.549961</td>
      <td>VeryHighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>6</th>
      <td>INTERCEPT</td>
      <td>-0.000606</td>
      <td>VeryHighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>0.233418</td>
      <td>VeryHighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GDPCAP</td>
      <td>0.508083</td>
      <td>VeryHighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>9</th>
      <td>INTERCEPT</td>
      <td>0.110662</td>
      <td>HighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LIFEXP</td>
      <td>0.317843</td>
      <td>HighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GDPCAP</td>
      <td>0.413052</td>
      <td>HighLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>12</th>
      <td>INTERCEPT</td>
      <td>-0.000585</td>
      <td>HighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LIFEXP</td>
      <td>0.237033</td>
      <td>HighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GDPCAP</td>
      <td>0.504420</td>
      <td>HighLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>15</th>
      <td>INTERCEPT</td>
      <td>2.133856</td>
      <td>LowLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>16</th>
      <td>LIFEXP</td>
      <td>1.811625</td>
      <td>LowLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GDPCAP</td>
      <td>1.831115</td>
      <td>LowLearningRate_LowEpochCount</td>
    </tr>
    <tr>
      <th>18</th>
      <td>INTERCEPT</td>
      <td>0.128061</td>
      <td>LowLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LIFEXP</td>
      <td>0.320584</td>
      <td>LowLearningRate_HighEpochCount</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GDPCAP</td>
      <td>0.415517</td>
      <td>LowLearningRate_HighEpochCount</td>
    </tr>
  </tbody>
</table>
</div>



```python
consolidated_ne = linear_regression_methods[linear_regression_methods['Method']=='Normal_Equations'].loc[:,"Estimate"]
consolidated_vhlr_lec = linear_regression_methods[linear_regression_methods['Method']=='VeryHighLearningRate_LowEpochCount'].loc[:,"Estimate"]
consolidated_vhlr_hec = linear_regression_methods[linear_regression_methods['Method']=='VeryHighLearningRate_HighEpochCount'].loc[:,"Estimate"]
consolidated_hlr_lec = linear_regression_methods[linear_regression_methods['Method']=='HighLearningRate_LowEpochCount'].loc[:,"Estimate"]
consolidated_hlr_hec = linear_regression_methods[linear_regression_methods['Method']=='HighLearningRate_HighEpochCount'].loc[:,"Estimate"]
consolidated_llr_lec = linear_regression_methods[linear_regression_methods['Method']=='LowLearningRate_LowEpochCount'].loc[:,"Estimate"]
consolidated_llr_hec = linear_regression_methods[linear_regression_methods['Method']=='LowLearningRate_HighEpochCount'].loc[:,"Estimate"]
linear_regression_methods_plot = pd.DataFrame({'NE': consolidated_ne.values,
                                               'VHLR_LEC': consolidated_vhlr_lec.values,
                                               'VHLR_HEC': consolidated_vhlr_hec.values,
                                               'HLR_LEC': consolidated_hlr_lec.values,
                                               'HLR_HEC': consolidated_hlr_hec.values,
                                               'LLR_LEC': consolidated_llr_lec.values,
                                               'LLR_HEC': consolidated_llr_hec.values},
                                              linear_regression_methods['Coefficient'].unique())
linear_regression_methods_plot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NE</th>
      <th>VHLR_LEC</th>
      <th>VHLR_HEC</th>
      <th>HLR_LEC</th>
      <th>HLR_HEC</th>
      <th>LLR_LEC</th>
      <th>LLR_HEC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>INTERCEPT</th>
      <td>-0.000606</td>
      <td>0.008008</td>
      <td>-0.000606</td>
      <td>0.110662</td>
      <td>-0.000585</td>
      <td>2.133856</td>
      <td>0.128061</td>
    </tr>
    <tr>
      <th>LIFEXP</th>
      <td>0.233418</td>
      <td>0.281761</td>
      <td>0.233418</td>
      <td>0.317843</td>
      <td>0.237033</td>
      <td>1.811625</td>
      <td>0.320584</td>
    </tr>
    <tr>
      <th>GDPCAP</th>
      <td>0.508083</td>
      <td>0.549961</td>
      <td>0.508083</td>
      <td>0.413052</td>
      <td>0.504420</td>
      <td>1.831115</td>
      <td>0.415517</td>
    </tr>
  </tbody>
</table>
</div>




```python
linear_regression_coefficent_estimation_methods_plot = linear_regression_methods_plot.plot.barh(figsize=(10, 6),width=0.90)
linear_regression_coefficent_estimation_methods_plot.set_xlim(-0.5,3)
linear_regression_coefficent_estimation_methods_plot.set_title("Linear Regression Coefficient Comparison by Estimation Method")
linear_regression_coefficent_estimation_methods_plot.set_xlabel("Linear Regression Coefficient Estimates")
linear_regression_coefficent_estimation_methods_plot.set_ylabel("Linear Regression Coefficients")
linear_regression_coefficent_estimation_methods_plot.grid(False)
linear_regression_coefficent_estimation_methods_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in linear_regression_coefficent_estimation_methods_plot.containers:
    linear_regression_coefficent_estimation_methods_plot.bar_label(container, fmt='%.5f', padding=5, color='black')
```


    
![png](output_202_0.png)
    


# 2. Summary <a class="anchor" id="Summary"></a>




# 3. References <a class="anchor" id="References"></a>

* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Book]** [Regression Modeling Strategies](https://link.springer.com/book/10.1007/978-1-4757-3462-1) by Frank Harrell
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model_selection](https://scikit-learn.org/stable/model_selection.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.pipeline](https://scikit-learn.org/stable/modules/compose.html) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Article]** [Step-by-Step Exploratory Data Analysis (EDA) using Python](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/#:~:text=Exploratory%20Data%20Analysis%20(EDA)%20with,distributions%20using%20Python%20programming%20language.) by Malamahadevan Mahadevan (Analytics Vidhya)
* **[Article]** [Exploratory Data Analysis in Python — A Step-by-Step Process](https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee) by Andrea D'Agostino (Towards Data Science)
* **[Article]** [Exploratory Data Analysis with Python](https://medium.com/@douglas.rochedo/exploratory-data-analysis-with-python-78b6c1d479cc) by Douglas Rocha (Medium)
* **[Article]** [4 Ways to Automate Exploratory Data Analysis (EDA) in Python](https://builtin.com/data-science/EDA-python) by Abdishakur Hassan (BuiltIn)
* **[Article]** [10 Things To Do When Conducting Your Exploratory Data Analysis (EDA)](https://www.analyticsvidhya.com) by Alifia Harmadi (Medium)
* **[Article]** [How to Handle Missing Data with Python](https://machinelearningmastery.com/handle-missing-data-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Imputing Missing Data with Simple and Advanced Techniques](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87) by Idil Ismiguzel (Towards Data Science)
* **[Article]** [Missing Data Imputation Approaches | How to handle missing values in Python](https://www.machinelearningplus.com/machine-learning/missing-data-imputation-how-to-handle-missing-values-in-python/) by Selva Prabhakaran (Machine Learning +)
* **[Article]** [Master The Skills Of Missing Data Imputation Techniques In Python(2022) And Be Successful](https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e) by Mrinal Walia (Analytics Vidhya)
* **[Article]** [How to Preprocess Data in Python](https://builtin.com/machine-learning/how-to-preprocess-data-python) by Afroz Chakure (BuiltIn)
* **[Article]** [Easy Guide To Data Preprocessing In Python](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html) by Ahmad Anis (KDNuggets)
* **[Article]** [Data Preprocessing in Python](https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5) by Tarun Gupta (Towards Data Science)
* **[Article]** [Data Preprocessing using Python](https://medium.com/@suneet.bhopal/data-preprocessing-using-python-1bfee9268fb3) by Suneet Jain (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.com/@abonia/data-preprocessing-in-python-1f90d95d44f4) by Abonia Sojasingarayar (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.datadriveninvestor.com/data-preprocessing-3cd01eefd438) by Afroz Chakure (Medium)
* **[Article]** [Detecting and Treating Outliers | Treating the Odd One Out!](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/) by Harika Bonthu (Analytics Vidhya)
* **[Article]** [Outlier Treatment with Python](https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02) by Sangita Yemulwar (Analytics Vidhya)
* **[Article]** [A Guide to Outlier Detection in Python](https://builtin.com/data-science/outlier-detection-python) by Sadrach Pierre (BuiltIn)
* **[Article]** [How To Find Outliers in Data Using Python (and How To Handle Them)](https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/) by Eric Kleppen (Career Foundry)
* **[Article]** [Statistics in Python — Collinearity and Multicollinearity](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) by Wei-Meng Lee (Towards Data Science)
* **[Article]** [Understanding Multicollinearity and How to Detect it in Python](https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc) by Terence Shin (Towards Data Science)
* **[Article]** [A Python Library to Remove Collinearity](https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/) by Gianluca Malato (Your Data Teacher)
* **[Article]** [8 Best Data Transformation in Pandas](https://ai.plainenglish.io/data-transformation-in-pandas-29b2b3c61b34) by Tirendaz AI (Medium)
* **[Article]** [Data Transformation Techniques with Python: Elevate Your Data Game!](https://medium.com/@siddharthverma.er.cse/data-transformation-techniques-with-python-elevate-your-data-game-21fcc7442cc2) by Siddharth Verma (Medium)
* **[Article]** [Data Scaling with Python](https://www.kdnuggets.com/2023/07/data-scaling-python.html) by Benjamin Obi Tayo (KDNuggets)
* **[Article]** [How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Feature Engineering: Scaling, Normalization, and Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/) by Aniruddha Bhandari  (Analytics Vidhya)
* **[Article]** [How to Normalize Data Using scikit-learn in Python](https://www.digitalocean.com/community/tutorials/normalize-data-in-python) by Jayant Verma (Digital Ocean)
* **[Article]** [What are Categorical Data Encoding Methods | Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/) by Shipra Saxena  (Analytics Vidhya)
* **[Article]** [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html) by Chris Moffitt (Practical Business Python)
* **[Article]** [Categorical Data Encoding Techniques in Python: A Complete Guide](https://soumenatta.medium.com/categorical-data-encoding-techniques-in-python-a-complete-guide-a913aae19a22) by Soumen Atta (Medium)
* **[Article]** [Categorical Feature Encoding Techniques](https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24) by Tara Boyle (Medium)
* **[Article]** [Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Hypothesis Testing with Python: Step by Step Hands-On Tutorial with Practical Examples](https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e) by Ece Işık Polat (Towards Data Science)
* **[Article]** [17 Statistical Hypothesis Tests in Python (Cheat Sheet)](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [A Step-by-Step Guide to Hypothesis Testing in Python using Scipy](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Gabriel Rennó (Medium)
* **[Article]** [Gradient Descent and Stochastic Gradient Descent in R](https://www.ocf.berkeley.edu/~janastas/stochastic-gradient-descent-in-r.html) by Jason Anastasopoulos
* **[Article]** [Linear Regression Tutorial Using Gradient Descent for Machine Learning](https://machinelearningmastery.com/linear-regression-tutorial-using-gradient-descent-for-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/) by Sebastian Ruder
* **[Article]** [What is Gradient Descent?](https://www.ibm.com/topics/gradient-descent) by IBM Team
* **[Article]** [Gradient Descent in Machine Learning: A Basic Introduction](https://builtin.com/data-science/gradient-descent) by Niklas Donges
* **[Article]** [Gradient Descent for Linear Regression Explained, Step by Step](https://machinelearningcompass.com/machine_learning_math/gradient_descent_for_linear_regression/) by Boris Giba
* **[Article]** [Gradient Descent Explained Simply with Examples](https://vitalflux.com/gradient-descent-explained-simply-with-examples/) by Ajitesh Kumar
* **[Article]** [Implementing the Gradient Descent Algorithm in R](https://www.r-bloggers.com/2017/02/implementing-the-gradient-descent-algorithm-in-r/) by Richter Walsh
* **[Article]** [Regression via Gradient Descent in R](https://www.r-bloggers.com/2011/11/regression-via-gradient-descent-in-r/) by Matt Bogard
* **[Article]** [Stochastic Gradient Descent](http://optimization.cbe.cornell.edu/index.php?title=Stochastic_gradient_descent) by Jonathon Price, Alfred Wong, Tiancheng Yuan, Joshua Mathew and Taiwo Olorunniwo
* **[Article]** [Difference Between Batch Gradient Descent and Stochastic Gradient Descent](https://www.geeksforgeeks.org/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/) by Geeks For Geeks Team
* **[Article]** [Stochastic Gradient Descent Vs Gradient Descent: A Head-To-Head Comparison](https://sdsclub.com/stochastic-gradient-descent-vs-gradient-descent-a-head-to-head-comparison/) by SDS Club Team
* **[Article]** [Differences Between Gradient, Stochastic and Mini Batch Gradient Descent](https://www.baeldung.com/cs/gradient-stochastic-and-mini-batch) by Baeldung
* **[Article]** [Difference Between Backpropagation and Stochastic Gradient Descent](https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [ML | Normal Equation in Linear Regression](https://www.geeksforgeeks.org/ml-normal-equation-in-linear-regression/) by Geeks For Geeks Team
* **[Article]** [Derivation of the Normal Equation for Linear Regression](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/) by Eli Bendersky
* **[Article]** [Normal Equation](http://mlwiki.org/index.php/Normal_Equation) by ML Wiki Team
* **[Article]** [Normal Equations](https://www.statlect.com/glossary/normal-equations#:~:text=In%20linear%20regression%20analysis%2C%20the%20normal%20equations%20are,first-order%20condition%20of%20the%20Least%20Squares%20minimization%20problem.) by Marco Taboga
* **[Article]** [Fitting a Model via Closed-form Equations versus Gradient Descent Versus Stochastic Gradient Descent Versus Mini-Batch Learning. What is the Difference?](https://sebastianraschka.com/faq/docs/closed-form-vs-gd.html) by Sebastian Raschka
* **[Article]** [Gradient Descent Versus Normal Equation For Regression Problems](https://pushkarasharma.github.io/2020/06/27/Gradient_vs_Normal.html) by Pushkara Sharma
* **[Article]** [Batch, Mini Batch & Stochastic Gradient Descent](https://towardsdatascience.com/batch-mini-batch-stochastic-gradient-descent-7a62ecba642a#:~:text=Batch%20Gradient%20Descent%20can%20be,converges%20faster%20for%20larger%20datasets.) by Sushant Patrikar (Towards Data Science)
* **[Article]** [Difference between Batch Gradient Descent and Stochastic Gradient Descent](https://www.geeksforgeeks.org/difference-between-batch-gradient-descent-and-stochastic-gradient-descent/) by Geeks for Geeks Team
* **[Article]** [Variants of Gradient Descent Algorithm](https://www.analyticsvidhya.com/blog/2021/03/variants-of-gradient-descent-algorithm/) by Himanshi Singh (Analytics Vidhya)
* **[Article]** [Difference Between SGD, GD, and Mini-batch GD](https://www.tutorialspoint.com/difference-between-sgd-gd-and-mini-batch-gd) by Jay Singh (TutorialsPoint)
* **[Article]** [Batch , Mini Batch and Stochastic Gradient Descent](https://sweta-nit.medium.com/batch-mini-batch-and-stochastic-gradient-descent-e9bc4cacd461) by Sweta (Mediuma)
* **[Article]** [A Gentle Introduction to Mini-Batch Gradient Descent and How to Configure Batch Size](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Batch vs Mini-batch vs Stochastic Gradient Descent with Code Examples](https://medium.datadriveninvestor.com/batch-vs-mini-batch-vs-stochastic-gradient-descent-with-code-examples-cd8232174e14) by Matheus Jacques (Medium)
* **[Article]** [Batch vs Stochastic vs Mini Batch Gradient Descent Techniques](https://medium.com/@amannagrawall002/batch-vs-stochastic-vs-mini-batch-gradient-descent-techniques-7dfe6f963a6f) by Aman Agrawal (Medium)
* **[Article]** [Difference Between a Batch and an Epoch in a Neural Network](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Batch, Mini-Batch and Stochastic Gradient Descent for Linear Regression](https://towardsdatascience.com/batch-mini-batch-and-stochastic-gradient-descent-for-linear-regression-9fe4eefa637c) by Robert Kwiatkowski (Towards Data Science)
* **[Publication]** [New Methods for the Determination of Comet Orbits](https://www.semanticscholar.org/paper/Nouvelles-m%C3%A9thodes-pour-la-d%C3%A9termination-des-des-Legendre/21c9090e226ab449ffb608ddb2cb925911a61f24) by Adrien-Marie Legendre
* **[Publication]** [General Method for the Resolution of a System of Simultaneous Equations](https://cs.uwaterloo.ca/~y328yu/classics/cauchy-en.pdf) by Augustine Cauchy
* **[Publication]** [Data Quality for Machine Learning Tasks](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Nitin Gupta, Shashank Mujumdar, Hima Patel, Satoshi Masuda, Naveen Panwar, Sambaran Bandyopadhyay, Sameep Mehta, Shanmukha Guttula, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Overview and Importance of Data Quality for Machine Learning Tasks](https://dl.acm.org/doi/10.1145/3394486.3406477) by Abhinav Jain, Hima Patel, Lokesh Nagalapatti, Nitin Gupta, Sameep Mehta, Shanmukha Guttula, Shashank Mujumdar, Shazia Afzal, Ruhi Sharma Mittal and Vitobha Munigala (KDD ’20: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining)
* **[Publication]** [Multiple Imputation of Discrete and Continuous Data by Fully Conditional Specification](https://journals.sagepub.com/doi/10.1177/0962280206074463) by Stef van Buuren (Statistical Methods in Medical Research)
* **[Publication]** [Mathematical Contributions to the Theory of Evolution: Regression, Heredity and Panmixia](https://royalsocietypublishing.org/doi/10.1098/rsta.1896.0007) by Karl Pearson (Royal Society)
* **[Publication]** [A New Family of Power Transformations to Improve Normality or Symmetry](https://academic.oup.com/biomet/article-abstract/87/4/954/232908?redirectedFrom=fulltext&login=false) by In-Kwon Yeo and Richard Johnson (Biometrika)
* **[Course]** [IBM Data Analyst Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-analyst) by IBM Team (Coursera)
* **[Course]** [IBM Data Science Professional Certificate](https://www.coursera.org/professional-certificates/ibm-data-science) by IBM Team (Coursera)
* **[Course]** [IBM Machine Learning Professional Certificate](https://www.coursera.org/professional-certificates/ibm-machine-learning) by IBM Team (Coursera)

***


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

