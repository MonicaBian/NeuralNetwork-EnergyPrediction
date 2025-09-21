#  Photovoltaic Energy Prediction using LSTM Neural Network Models

Neural networks play a crucial role in predicting photovoltaic (PV) energy data on solar inverters because of their ability to capture the complex, nonlinear relationships between weather conditions, solar irradiance, and system performance. 

Unlike traditional statistical methods, neural networks can learn from large volumes of historical inverter and meteorological data to provide highly accurate short- and long-term forecasts of energy generation.

These predictions are essential for optimizing inverter operations, ensuring grid stability, and minimizing energy curtailment during periods of high solar penetration. 

By accurately forecasting PV output, neural networks support better integration of renewable energy into the grid, enhance the reliability of solar power systems, and contribute to more efficient energy management and planning.


# Notebook Summary

This notebook:
1. Trains an LSTM model on **January–November 2023** photovoltaic data.
2. Validates with the tail of that training window (chronological split).
3. Generates Power predictions for **December 2023** and computes metrics.
4. Saves artifacts (model + scalers + config) and CSV outputs.


# Visual Comparison

Below are sample outputs that demonstrate the value of dynamic neural network predictions. We are able to accurately predict the photovoltaic energy absorbed by inverters throughout the day with small variances. This is useful as sudden disruptions or unexpected anti-islanding will display a signficant outlier in the graph.

![Power vs GHI on 2023-12-03](https://github.com/MonicaBian/NeuralNetwork-EnergyPrediction/blob/main/Images/power_ghi_2023-12-07.png)


#  Quantitative Analysis 
To support our results, we provide a quantitative view that measures the accuracy of our predictions against the known ground truth for the Decemeber dataset.

Errors are slightly higher during daylight (MAE_day ≈ 0.19 vs. MAE ≈ 0.14).
This makes sense: at night, power is zero and trivial to predict, while daylight introduces variability from weather/clouds.

- MAE & MSE & RMSE are low → predictions are generally close in absolute terms.
- WAPE ~18% → forecasts are moderately accurate; good but not best-in-class.
- High R² (0.9+) → strong explanatory power, capturing most variability.