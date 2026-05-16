# A02-sam19028-utk25002

Hello! It's Samra Mujcinovic and Oceane Pacode - welcome to our repo! 

We will train a regression model on the California Housing dataset. 

---

## 📌 Overview

This project uses an MLPRegressor neural network model on the California Housing dataset using scikit-learn to predict median house value. 

### 🔄 Workflow
- loading the dataset
- train/test splitting
- feature scaling
- training the model
- generating actual vs predicted scatter plots
- evaluating performance using RMSE, MAE, R², and MAPE

---

## 📂 Files

- `src/model.py` → main training and evaluation script
- `figures/train_actual_vs_pred.png` → training set scatter plot
- `figures/test_actual_vs_pred.png` → test set scatter plot
- `requirements.txt` → required Python packages

---

## ⚙️ Required Packages & Running the Project

Install packages using:

```bash
pip install -r requirements.txt
```
Run the model with:

```bash
python src/model.py
```
---

## 📊 Expected Output

Running the script will:
- print training and test evaluation metrics (RMSE, MAE, R², and MAPE)
- save scatter plots comparing actual vs predicted house values

Generated figures:
- `figures/train_actual_vs_pred.png`
- `figures/test_actual_vs_pred.png`

---

## 👥 Authors

- Samra Mujcinovic
- Oceane Pacode
