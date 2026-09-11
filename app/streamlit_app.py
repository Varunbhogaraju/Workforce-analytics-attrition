import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Workforce Analytics",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "employee_attrition.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "reports",
    "models",
    "best_attrition_model.joblib"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

df["AttritionFlag"] = df["Attrition"].map(
    {"Yes": 1, "No": 0}
)

feature_df = df.drop(
    columns=["Attrition", "AttritionFlag"]
)

attrition_rate = (
    df["AttritionFlag"].mean() * 100
)

st.sidebar.title("Workforce Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Workforce Explorer",
        "Attrition Prediction",
        "Model Lab"
    ]
)

st.sidebar.divider()

st.sidebar.write("Dataset")
st.sidebar.write(f"{len(df):,} employees")
st.sidebar.write(f"{len(feature_df.columns)} features")


if page == "Executive Dashboard":

    st.title("📊 Workforce Analytics Dashboard")

    st.caption(
        "Employee workforce analysis and attrition risk insights"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Employees",
        f"{len(df):,}"
    )

    c2.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    c3.metric(
        "Employees Left",
        f"{int(df['AttritionFlag'].sum()):,}"
    )

    c4.metric(
        "Average Tenure",
        f"{df['YearsAtCompany'].mean():.1f} years"
    )

    st.divider()

    col1, col2 = st.columns(2)

    department_attrition = (
        df.groupby("Department")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    overtime_attrition = (
        df.groupby("OverTime")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    with col1:

        fig = px.bar(
            department_attrition,
            x="Department",
            y="AttritionRate",
            title="Attrition Rate by Department",
            labels={
                "AttritionRate": "Attrition Rate (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            overtime_attrition,
            x="OverTime",
            y="AttritionRate",
            title="Attrition Rate by Overtime",
            labels={
                "AttritionRate": "Attrition Rate (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    satisfaction_attrition = (
        df.groupby("JobSatisfaction")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    income_data = df.copy()

    income_data["Status"] = income_data["Attrition"].map(
        {
            "Yes": "Employees Who Left",
            "No": "Employees Who Stayed"
        }
    )

    with col1:

        fig = px.line(
            satisfaction_attrition,
            x="JobSatisfaction",
            y="AttritionRate",
            markers=True,
            title="Attrition vs Job Satisfaction",
            labels={
                "AttritionRate": "Attrition Rate (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.box(
            income_data,
            x="Status",
            y="MonthlyIncome",
            title="Monthly Income by Attrition Status",
            labels={
                "MonthlyIncome": "Monthly Income"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Key Findings")

    st.write(
        f"• Overall attrition rate is {attrition_rate:.2f}%."
    )

    st.write(
        "• Overtime shows a strong relationship with employee attrition."
    )

    st.write(
        "• Job satisfaction is associated with differences in attrition rates."
    )

    st.write(
        "• Employees who left have different income distributions from employees who stayed."
    )

    st.warning(
        "Attrition represents only 16.12% of observations, so accuracy alone "
        "is not sufficient for evaluating the classification model."
    )


elif page == "Workforce Explorer":

    st.title("🔎 Workforce Explorer")

    col1, col2 = st.columns(2)

    with col1:

        department_options = [
            "All"
        ] + sorted(
            df["Department"].unique().tolist()
        )

        selected_department = st.selectbox(
            "Department",
            department_options
        )

    with col2:

        selected_attrition = st.selectbox(
            "Attrition",
            ["All", "Yes", "No"]
        )

    filtered = df.copy()

    if selected_department != "All":

        filtered = filtered[
            filtered["Department"] == selected_department
        ]

    if selected_attrition != "All":

        filtered = filtered[
            filtered["Attrition"] == selected_attrition
        ]

    st.metric(
        "Filtered Employees",
        f"{len(filtered):,}"
    )

    st.divider()

    col1, col2 = st.columns(2)

    role_data = (
        filtered.groupby("JobRole")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    education_data = (
        filtered.groupby("EducationField")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    with col1:

        fig = px.bar(
            role_data,
            x="JobRole",
            y="AttritionRate",
            title="Attrition by Job Role"
        )

        fig.update_xaxes(
            tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            education_data,
            x="EducationField",
            y="AttritionRate",
            title="Attrition by Education Field"
        )

        fig.update_xaxes(
            tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col1, col2 = st.columns(2)

    marital_data = (
        filtered.groupby("MaritalStatus")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    travel_data = (
        filtered.groupby("BusinessTravel")["AttritionFlag"]
        .mean()
        .mul(100)
        .reset_index(name="AttritionRate")
    )

    with col1:

        fig = px.bar(
            marital_data,
            x="MaritalStatus",
            y="AttritionRate",
            title="Attrition by Marital Status"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            travel_data,
            x="BusinessTravel",
            y="AttritionRate",
            title="Attrition by Business Travel"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Numeric Feature Distributions")

    numeric_features = [
        "Age",
        "MonthlyIncome",
        "YearsAtCompany",
        "DistanceFromHome",
        "JobLevel",
        "YearsInCurrentRole"
    ]

    selected_feature = st.selectbox(
        "Select feature",
        numeric_features
    )

    fig = px.box(
        filtered,
        x="Attrition",
        y=selected_feature,
        title=f"{selected_feature} by Attrition"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Filtered Employee Data")

    columns_to_display = [
        "Age",
        "Department",
        "JobRole",
        "MonthlyIncome",
        "YearsAtCompany",
        "JobSatisfaction",
        "OverTime",
        "Attrition"
    ]

    st.dataframe(
        filtered[columns_to_display],
        use_container_width=True
    )


elif page == "Attrition Prediction":

    st.title("🎯 Employee Attrition Prediction")

    st.write(
        "Estimate attrition risk using the trained machine-learning model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            18,
            70,
            30
        )

        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            50000,
            5000
        )

        years_at_company = st.number_input(
            "Years at Company",
            0,
            50,
            5
        )

        distance_from_home = st.number_input(
            "Distance From Home",
            1,
            100,
            5
        )

        job_level = st.number_input(
            "Job Level",
            1,
            5,
            2
        )

    with col2:

        department = st.selectbox(
            "Department",
            sorted(df["Department"].unique())
        )

        job_role = st.selectbox(
            "Job Role",
            sorted(df["JobRole"].unique())
        )

        overtime = st.selectbox(
            "OverTime",
            sorted(df["OverTime"].unique())
        )

        business_travel = st.selectbox(
            "Business Travel",
            sorted(df["BusinessTravel"].unique())
        )

        marital_status = st.selectbox(
            "Marital Status",
            sorted(df["MaritalStatus"].unique())
        )

    with col3:

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            4,
            3
        )

        work_life_balance = st.slider(
            "Work Life Balance",
            1,
            4,
            3
        )

        environment_satisfaction = st.slider(
            "Environment Satisfaction",
            1,
            4,
            3
        )

        stock_option_level = st.slider(
            "Stock Option Level",
            0,
            3,
            1
        )

        num_companies = st.number_input(
            "Number of Companies Worked",
            0,
            20,
            2
        )

    threshold = st.slider(
        "Classification Threshold",
        0.10,
        0.90,
        0.50,
        0.05
    )

    if st.button(
        "Calculate Attrition Risk",
        type="primary",
        use_container_width=True
    ):

        input_data = {}

        for column in feature_df.columns:

            if pd.api.types.is_numeric_dtype(
                feature_df[column]
            ):

                input_data[column] = (
                    feature_df[column].median()
                )

            else:

                input_data[column] = (
                    feature_df[column].mode()[0]
                )

        input_data["Age"] = age
        input_data["MonthlyIncome"] = monthly_income
        input_data["YearsAtCompany"] = years_at_company
        input_data["DistanceFromHome"] = distance_from_home
        input_data["JobLevel"] = job_level
        input_data["Department"] = department
        input_data["JobRole"] = job_role
        input_data["OverTime"] = overtime
        input_data["BusinessTravel"] = business_travel
        input_data["MaritalStatus"] = marital_status
        input_data["JobSatisfaction"] = job_satisfaction
        input_data["WorkLifeBalance"] = work_life_balance
        input_data["EnvironmentSatisfaction"] = environment_satisfaction
        input_data["StockOptionLevel"] = stock_option_level
        input_data["NumCompaniesWorked"] = num_companies

        input_df = pd.DataFrame(
            [input_data]
        )

        expected_columns = getattr(
            model,
            "feature_names_in_",
            feature_df.columns
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in input_df.columns
        ]

        if missing_columns:

            st.error(
                "Model expects missing features: "
                + ", ".join(missing_columns)
            )

        else:

            input_df = input_df[
                list(expected_columns)
            ]

            try:

                probability = model.predict_proba(
                    input_df
                )[0][1]

                prediction = probability >= threshold

                st.divider()

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Attrition Probability",
                    f"{probability:.1%}"
                )

                c2.metric(
                    "Threshold",
                    f"{threshold:.0%}"
                )

                c3.metric(
                    "Classification",
                    "High Risk"
                    if prediction
                    else "Lower Risk"
                )

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={
                            "text": "Estimated Attrition Risk"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            }
                        }
                    )
                )

                gauge.update_layout(
                    height=350
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )

                if prediction:

                    st.error(
                        "The model classifies this employee as higher attrition risk."
                    )

                else:

                    st.success(
                        "The model classifies this employee as lower attrition risk."
                    )

                st.info(
                    "This is a statistical risk estimate based on historical "
                    "patterns and should not be treated as a definitive judgment."
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


elif page == "Model Lab":

    st.title("🧪 Model Lab")

    st.subheader("Model Comparison")

    model_results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Random Forest"
            ],
            "Accuracy": [
                0.751701,
                0.826531
            ],
            "Precision": [
                0.341463,
                0.437500
            ],
            "Recall": [
                0.595745,
                0.297872
            ],
            "F1": [
                0.434109,
                0.354430
            ],
            "ROC-AUC": [
                0.807908,
                0.792532
            ]
        }
    )

    st.dataframe(
        model_results.style.format(
            {
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            }
        ),
        use_container_width=True
    )

    fig = px.bar(
        model_results,
        x="Model",
        y=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ],
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Statistical Tests")

    statistical_tests = pd.DataFrame(
        {
            "Test": [
                "Monthly Income t-test",
                "Overtime Chi-square"
            ],
            "Statistic": [
                -7.4826215866,
                87.5642936583
            ],
            "p-value": [
                4.4335886283e-13,
                8.1584237215e-21
            ]
        }
    )

    st.dataframe(
        statistical_tests.style.format(
            {
                "Statistic": "{:.4f}",
                "p-value": "{:.3e}"
            }
        ),
        use_container_width=True
    )

    st.warning(
        "Statistical significance does not establish causation."
    )

    st.subheader("Cross-Validation")

    c1, c2 = st.columns(2)

    c1.metric(
        "5-Fold CV F1",
        "0.488"
    )

    c2.metric(
        "CV Standard Deviation",
        "0.015"
    )

    st.subheader("Model Selection")

    st.success(
        "Logistic Regression was selected because it achieved higher "
        "F1 and recall for the minority attrition class, with ROC-AUC "
        "of approximately 0.81."
    )

    importance_path = os.path.join(
        BASE_DIR,
        "reports",
        "feature_importance.csv"
    )

    if os.path.exists(importance_path):

        st.subheader("Feature Importance")

        importance = pd.read_csv(
            importance_path
        )

        if len(importance.columns) >= 2:

            feature_column = importance.columns[0]
            importance_column = importance.columns[1]

            importance = (
                importance
                .sort_values(
                    importance_column,
                    ascending=False
                )
                .head(15)
            )

            fig = px.bar(
                importance.sort_values(
                    importance_column
                ),
                x=importance_column,
                y=feature_column,
                orientation="h",
                title="Top Features"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance,
                use_container_width=True
            )

    st.subheader("Interpretation")

    st.write(
        "The dataset contains 1,470 employees, with attrition representing "
        "16.12% of observations. Because of this class imbalance, F1-score, "
        "recall and ROC-AUC are more informative than accuracy alone."
    )