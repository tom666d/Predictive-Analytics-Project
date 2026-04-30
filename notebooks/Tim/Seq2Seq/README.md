# GRU Seq2Seq Forecasting Experiment

This folder contains Tim's exploratory sequence-based forecasting experiments for the M5 Forecasting Accuracy competition.

Goal:
- Build a lightweight GRU-based sequence forecasting model
- Use past sales sequences to predict the next 28 days
- Generate a valid full Kaggle submission
- Compare the result with tree-based models and moving average baselines

Initial setup:
- Input: past 90 days of sales
- Output: next 28 days of sales
- Transform: log1p(sales)
- Model: GRU sequence forecasting model