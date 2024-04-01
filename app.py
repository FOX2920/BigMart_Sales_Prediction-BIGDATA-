import streamlit as st
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Streamlit App") \
    .getOrCreate()

# Load trained models
gbt_model = PipelineModel.load('models/GBTRegressor')
lr_model = PipelineModel.load('models/LinearRegression')
dt_model = PipelineModel.load('models/DecisionTreeRegressor')

# Define Streamlit app
def main():
    st.title('Big Mart Sales Prediction')
    st.sidebar.title('Model Selection')

    # Sidebar for model selection
    selected_model = st.sidebar.selectbox(
        'Select Model',
        ('Gradient Boosted Trees', 'Linear Regression', 'Decision Tree')
    )

    # Function to make predictions
    def predict_sales(data):
        if selected_model == 'Gradient Boosted Trees':
            prediction = gbt_model.transform(data)
        elif selected_model == 'Linear Regression':
            prediction = lr_model.transform(data)
        elif selected_model == 'Decision Tree':
            prediction = dt_model.transform(data)
        return prediction

    # File uploader for test data
    uploaded_file = st.file_uploader("Upload Test Data (CSV)", type=['csv'])
    if uploaded_file is not None:
        test_df = pd.read_csv(uploaded_file)
        spark_df = spark.createDataFrame(test_df)

        # Make predictions
        predictions = predict_sales(spark_df)

        # Display predictions
        st.write(predictions.toPandas())

if __name__ == '__main__':
    main()
