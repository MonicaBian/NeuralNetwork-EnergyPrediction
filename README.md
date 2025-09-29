#  Time-Series Modeling and Prediction of Photovoltaic Energy using Machine-Learning Models

Neural networks play a crucial role in predicting photovoltaic (PV) energy data on solar inverters because of their ability to capture the complex, nonlinear relationships between weather conditions, solar irradiance, and system performance. Unlike traditional statistical methods, neural networks can learn from large volumes of historical inverter and meteorological data to provide highly accurate short- and long-term forecasts of energy generation.

These predictions are essential for optimizing inverter operations, ensuring grid stability, and minimizing energy curtailment during periods of high solar penetration. By accurately forecasting PV output, neural networks support better integration of renewable energy into the grid, enhance the reliability of solar power systems, and contribute to more efficient energy management and planning.

This project applies Large Language Models (LLMs) and Neural Network models, such as Long Short-Term Memory (LSTM), to predict and forecast solar generation time-series data for curtailment detection and quantification. The machine-learning models were trained using a set of input variables that capture both temporal and environmental conditions: surface global irradiance, direct normal irradiance, surface diffuse irradiance, cloud type, cloud optical depth, solar elevation, and solar azimuth.

# Notebook Summary

This notebook:
1. Trains an LSTM model on **January–November 2023** photovoltaic data.
2. Validates with the tail of that training window (chronological split).
3. Generates power predictions for **December 2023** and computes metrics.
4. Saves artifacts (model + scalers + config) and CSV outputs.
5. Parses the CSV results, visualizes and plots the CSV outputs.



# Visual Comparison

The examples below highlight the effectiveness of the dynamic neural network predictions. The model accurately forecasts solar generation throughout the day with minimal variance. This capability is valuable, as sudden disruptions or unexpected anti-islanding events appear as pronounced outliers in the plotted results.  

![Power vs GHI on 2023-12-03](https://github.com/MonicaBian/NeuralNetwork-EnergyPrediction/blob/main/Images/power_ghi_2023-12-07.png)
![Power vs GHI on 2023-12-03](https://github.com/MonicaBian/NeuralNetwork-EnergyPrediction/blob/main/Images/power_voltage_2023-12-07.png)

#  Qualitative and Quantitative Analysis
To substantiate our findings, we present a quantitative evaluation of prediction accuracy against the ground-truth data for the December dataset.

Prediction errors are slightly higher during daylight hours (MAE_day ≈ 0.19 vs. MAE ≈ 0.14). This is expected: at night, power generation is zero and thus trivial to predict, whereas daylight introduces variability due to weather and cloud conditions.

Performance highlights:

- Low **MAE, MSE, RMSE** → predictions are close to actual outputs.
- **WAPE ≈ 18%** → good overall accuracy.
- **R² > 0.9** → strong fit, capturing most of the variation in the data.


___

# Description of the five statistical accuracy metrics applied to assess prediction performance.

## **Mean Absolute Error (MAE)**
- **Definition:** Average absolute difference between predicted and actual values.
- <span style="color:green"> **Good prediction:** </span> Small MAE (close to 0) → on average, predictions deviate only slightly from actual power.
- <span style="color:red"> **Bad prediction:** </span> Large MAE (no upper bound) → predictions consistently miss the target by a wide margin.
- **Scale:** Interpreted in Watts


## **Mean Squared Error (MSE)**
- **Definition:** Average of squared differences between predicted and actual values. Penalizes larger errors more than smaller ones.
- <span style="color:green"> **Good prediction:** </span> Low MSE → few large deviations.
- <span style="color:red"> **Bad prediction:** </span> High MSE → presence of large outliers/errors.
- **Scale:** In squared units (Watts²), so less intuitive than MAE.

## **Root Mean Squared Error (RMSE)**

- **Definition:** Square root of MSE; same unit as target variable.
- <span style="color:green"> **Good prediction:** </span> Low RMSE → model tracks actual data well with few large errors.
- <span style="color:red"> **Bad prediction:** </span> High RMSE → predictions have large swings away from truth.
- **Interpretation:** RMSE ≥ MAE usually; if RMSE is much larger than MAE, it means some large outliers dominate error.



## **Weighted Absolute Percentage Error (WAPE)**

- **Definition:** Total absolute error divided by total actuals, expressed as a percentage.
- <span style="color:green"> **Good prediction:** </span> WAPE close to 0%.
Rule of thumb thresholds:
    - <10% → very good
    - 10–20% → acceptable
    - 20–50% → weak but sometimes tolerable
    - 50% or above → poor
- <span style="color:red"> **Bad prediction:** </span> High WAPE suggests systematic deviation when compared to the magnitude of total production.


## **R² (R-squared, Coefficient of Determination)**
- **Definition:** Proportion of variance in the actual data explained by the predictions (0–1 scale, sometimes negative if worse than a baseline).
- <span style="color:green"> **Good prediction:** </span> 
    - R² close to 1.0 → model explains almost all variability in actual power.
    - R² ≥ 0.9 → excellent; 
    - R² = 0.7–0.9 → good.
- <span style="color:red"> **Bad prediction:** </span> 
    - R² close to 0 → model does not explain variability.
    - Negative R² → model performs worse than simply predicting the mean of actuals.
**Scale:** Dimensionless, bounded between (−∞, 1].


___

The December results are presented below:

| Date       | MAE    | MSE    | RMSE   | WAPE % | R²    | N   | MAE (Day) | MSE (Day) | RMSE (Day) | WAPE Day % | R² Day | N Day |
|------------|--------|--------|--------|--------|-------|-----|-----------|-----------|------------|-------------|--------|-------|
| 2023-12-01 | 0.149  | 0.075  | 0.274  | 11.83  | 0.958 | 282 | 0.193     | 0.098     | 0.314      | 11.72       | 0.942  | 216   |
| 2023-12-02 | 0.167  | 0.097  | 0.311  | 15.75  | 0.931 | 288 | 0.218     | 0.129     | 0.359      | 15.42       | 0.905  | 216   |
| 2023-12-03 | 0.072  | 0.013  | 0.116  |  5.27  | 0.994 | 288 | 0.088     | 0.017     | 0.131      |  5.07       | 0.993  | 227   |
| 2023-12-04 | 0.085  | 0.013  | 0.115  |  6.29  | 0.994 | 288 | 0.100     | 0.016     | 0.126      |  6.19       | 0.993  | 242   |
| 2023-12-05 | 0.161  | 0.073  | 0.270  | 20.72  | 0.922 | 288 | 0.184     | 0.084     | 0.290      | 20.65       | 0.914  | 251   |
| 2023-12-06 | 0.168  | 0.103  | 0.321  | 17.03  | 0.941 | 288 | 0.223     | 0.139     | 0.373      | 16.76       | 0.927  | 213   |
| 2023-12-07 | 0.082  | 0.017  | 0.130  | 16.25  | 0.954 | 288 | 0.128     | 0.028     | 0.168      | 14.91       | 0.910  | 170   |
| 2023-12-08 | 0.150  | 0.067  | 0.260  | 39.36  | 0.778 | 288 | 0.241     | 0.114     | 0.337      | 37.35       | 0.667  | 170   |
| 2023-12-09 | 0.101  | 0.026  | 0.162  | 49.29  | 0.591 | 288 | 0.166     | 0.045     | 0.211      | 48.26       | 0.251  | 170   |
| 2023-12-10 | 0.100  | 0.026  | 0.160  | 30.87  | 0.890 | 288 | 0.164     | 0.043     | 0.208      | 30.09       | 0.840  | 170   |
| 2023-12-11 | 0.156  | 0.078  | 0.279  | 32.88  | 0.876 | 288 | 0.259     | 0.132     | 0.363      | 32.44       | 0.835  | 170   |
| 2023-12-12 | 0.209  | 0.134  | 0.365  | 20.07  | 0.926 | 288 | 0.345     | 0.226     | 0.475      | 19.50       | 0.872  | 170   |
| 2023-12-13 | 0.159  | 0.085  | 0.291  | 20.36  | 0.888 | 288 | 0.227     | 0.125     | 0.354      | 19.74       | 0.819  | 195   |
| 2023-12-14 | 0.183  | 0.111  | 0.334  | 20.00  | 0.896 | 288 | 0.250     | 0.154     | 0.392      | 19.74       | 0.850  | 208   |
| 2023-12-15 | 0.201  | 0.150  | 0.387  | 15.30  | 0.936 | 288 | 0.264     | 0.200     | 0.447      | 15.04       | 0.914  | 216   |
| 2023-12-16 | 0.252  | 0.212  | 0.461  | 21.46  | 0.890 | 288 | 0.328     | 0.278     | 0.527      | 21.33       | 0.859  | 220   |
| 2023-12-17 | 0.114  | 0.064  | 0.254  |  8.73  | 0.971 | 288 | 0.144     | 0.083     | 0.288      |  8.51       | 0.963  | 223   |
| 2023-12-18 | 0.164  | 0.082  | 0.286  | 26.93  | 0.908 | 288 | 0.205     | 0.104     | 0.322      | 26.55       | 0.896  | 227   |
| 2023-12-19 | 0.095  | 0.032  | 0.178  |  6.84  | 0.987 | 288 | 0.139     | 0.049     | 0.222      |  6.43       | 0.978  | 184   |
| 2023-12-20 | 0.051  | 0.005  | 0.070  |  3.39  | 0.998 | 288 | 0.072     | 0.008     | 0.089      |  2.88       | 0.996  | 172   |
| 2023-12-21 | 0.085  | 0.025  | 0.159  |  6.22  | 0.990 | 288 | 0.129     | 0.042     | 0.205      |  5.66       | 0.980  | 172   |
| 2023-12-22 | 0.148  | 0.075  | 0.273  | 12.39  | 0.957 | 288 | 0.235     | 0.125     | 0.353      | 11.77       | 0.904  | 172   |
| 2023-12-23 | 0.096  | 0.017  | 0.129  |  6.93  | 0.992 | 288 | 0.114     | 0.020     | 0.143      |  6.63       | 0.991  | 232   |
| 2023-12-24 | 0.226  | 0.175  | 0.418  | 22.71  | 0.875 | 288 | 0.275     | 0.217     | 0.466      | 22.30       | 0.849  | 232   |
| 2023-12-25 | 0.137  | 0.052  | 0.228  | 30.19  | 0.874 | 288 | 0.167     | 0.065     | 0.255      | 29.57       | 0.857  | 231   |
| 2023-12-26 | 0.158  | 0.079  | 0.281  | 21.08  | 0.903 | 288 | 0.192     | 0.099     | 0.314      | 20.55       | 0.883  | 230   |
| 2023-12-27 | 0.184  | 0.140  | 0.375  | 15.74  | 0.933 | 284 | 0.234     | 0.184     | 0.429      | 15.32       | 0.916  | 217   |
| 2023-12-28 | 0.169  | 0.116  | 0.340  | 18.96  | 0.928 | 288 | 0.217     | 0.153     | 0.391      | 18.38       | 0.915  | 218   |
| 2023-12-29 | 0.098  | 0.027  | 0.165  |  7.02  | 0.988 | 288 | 0.131     | 0.039     | 0.196      |  6.60       | 0.981  | 202   |
| 2023-12-30 | 0.163  | 0.088  | 0.296  | 17.03  | 0.942 | 288 | 0.267     | 0.148     | 0.385      | 16.51       | 0.901  | 170   |
| 2023-12-31 | 0.179  | 0.186  | 0.431  | 15.39  | 0.924 | 287 | 0.291     | 0.315     | 0.561      | 14.75       | 0.87