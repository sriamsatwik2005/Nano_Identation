Your README looks great and clear! Here’s a polished version including the **virtual environment** note (which is good practice to mention) and some small formatting tweaks for clarity and consistency:

````markdown
# Nano Indentation Prediction

This project predicts nanoindentation depth of materials using machine learning. It uses a CatBoost regression model trained on laser and loading parameters to estimate indentation depth, helping analyze material deformation without physical testing.

## Project Files

- `app.py`: Streamlit app for real-time prediction  
- `catboost_model.cbm`: Trained CatBoost model  
- `train_model.py`: Script to train the model  
- `evaluate_model.py`: Script to evaluate the model  
- `NewTrainData.csv`: Training dataset  
- `NewTestData.csv`: Test dataset  
- `requirements.txt`: Python dependencies  

## How to Use

1. Clone the repo  
2. (Optional but recommended) Create and activate a Python virtual environment:
   - macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
````

4. Train the model:

   ```bash
   python train_model.py
   ```
5. Evaluate the model:

   ```bash
   python evaluate_model.py
   ```
6. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Input Features

* Laser pulse duration (ns)
* Laser energy (mJ)
* Loading rate (µN/s)
* Load (µN)
* Time (s)

## Output

The model predicts indentation depth in nanometers.
