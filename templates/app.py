from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# ==========================================
# CREATE FLASK APPLICATION
# ==========================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

app = Flask(__name__, template_folder=str(APP_DIR))


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(PROJECT_DIR / "model" / "employee_attrition_model.pkl")

NUMERIC_FIELDS = {
    "Age", "Job_Level", "Monthly_Income", "Hourly_Rate",
    "Years_at_Company", "Years_in_Current_Role",
    "Years_Since_Last_Promotion", "Work_Life_Balance", "Job_Satisfaction",
    "Performance_Rating", "Training_Hours_Last_Year", "Project_Count",
    "Average_Hours_Worked_Per_Week", "Absenteeism",
    "Work_Environment_Satisfaction", "Relationship_with_Manager",
    "Job_Involvement", "Distance_From_Home", "Number_of_Companies_Worked",
}

FORM_FIELDS = [
    "Age", "Gender", "Marital_Status", "Department", "Job_Role", "Job_Level",
    "Monthly_Income", "Hourly_Rate", "Years_at_Company",
    "Years_in_Current_Role", "Years_Since_Last_Promotion", "Work_Life_Balance",
    "Job_Satisfaction", "Performance_Rating", "Training_Hours_Last_Year",
    "Overtime", "Project_Count", "Average_Hours_Worked_Per_Week",
    "Absenteeism", "Work_Environment_Satisfaction",
    "Relationship_with_Manager", "Job_Involvement", "Distance_From_Home",
    "Number_of_Companies_Worked",
]


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html", form_data={})


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    try:
        data = {
            field: int(form_data[field]) if field in NUMERIC_FIELDS else form_data[field]
            for field in FORM_FIELDS
        }
        input_data = pd.DataFrame([data])
        prediction = int(model.predict(input_data)[0])
        probability = round(float(model.predict_proba(input_data)[0][1]) * 100, 2)
        return render_template(
            "index.html",
            prediction=("Employee likely to leave" if prediction else "Employee likely to stay"),
            probability=probability,
            result_class="danger" if prediction else "success",
            form_data=data
        )
    except (KeyError, TypeError, ValueError) as error:
        return render_template("index.html", error=f"Please check the submitted values: {error}", form_data=form_data), 400
    except Exception as error:
        return render_template("index.html", error=f"The prediction could not be completed: {error}", form_data=form_data), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)