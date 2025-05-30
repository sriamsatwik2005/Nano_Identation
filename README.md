# Nano Indentation Prediction
This project predicts nanoindentation depth of materials using machine learning. It uses a CatBoost regression model trained on laser and loading parameters to estimate indentation depth, helping analyze material deformation without physical testing.

## Project Files

- app.py: Streamlit app for real-time prediction  
- catboost_model.cbm: Trained CatBoost model  
- train_model.py: Script to train the model  
- evaluate_model.py: Script to evaluate the model  
- NewTrainData.csv: Training dataset  
- NewTestData.csv: Test dataset  
- requirements.txt: Python dependencies  

## How to Use

1. Clone the repo  
2. Install dependencies: `pip install -r requirements.txt`  
3. Train model: `python train_model.py`  
4. Evaluate model: `python evaluate_model.py`  
5. Run app: `streamlit run app.py`  

## Input Features

- Laser pulse duration (ns)  
- Laser energy (mJ)  
- Loading rate (µN/s)  
- Load (µN)  
- Time (s)  

## Output
The model predicts indentation depth in nanometers.
