# Workforce Analytics — Run Guide

Place `employee_attrition.csv` in `data/raw/`.

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\train_pipeline.py
streamlit run app\streamlit_app.py
```

The app contains five sections: Executive Dashboard, Workforce Explorer, Attrition Prediction, Model Lab, and GenAI Feedback.
