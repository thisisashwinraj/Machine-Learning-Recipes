import joblib
import pickle
import time

import pandas as pd
import numpy as np
import streamlit as st

from regression.linear_regression import LinearRegressionModel


# Set the title and favicon for the streamlit web application
st.set_page_config(
    page_title="Machine Learning Recipes",
    page_icon="assets/logo/ml_recipes_logo.png",
)


# Remove the extra padding from the top margin of the web app
st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 2rem;
					padding-bottom: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

# Hide streamlit menu and footer from the web app's interface
hide_menu_style = """
<style>
#MainMenu  {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Hide streamlit's default image expanders from app interface
hide_img_fs = """
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
"""
st.markdown(hide_img_fs, unsafe_allow_html=True)


types_of_ml_algorithms = {
    'Simple Linear Regression': 'Simple Linear Regression', 
    #'Multivariate Regression': 'Multivariate Regression',
    #'Decision Tree Regression': 'Decision Tree Regression',
    #'Random Forest Regression': 'Random Forest Regression',
    #'SVM Based Regression': 'SVM Based Regression',
}

selected_algorithm = st.sidebar.selectbox("Select the Regression Algorithm", types_of_ml_algorithms)

if selected_algorithm == 'Simple Linear Regression':
    model_filename = 'inference/linear_regression_model.pkl'
    model = joblib.load(model_filename)

    linear_regression_code = """
    model = LinearRegression()
    model.fit(X_train_scaled,y_train)
    """

    st.sidebar.code(linear_regression_code, language='python')
    st.sidebar.markdown("<br>", unsafe_allow_html=True)

    my_expander = st.sidebar.expander("Simple Linear Regression Plot", expanded=True)
    with my_expander:
        st.pyplot(LinearRegressionModel.plot_static_regression_line('datasets/diamonds.csv', 'carat'))

    st.title('Diamond Price Prediction :gem:')

    st.markdown("<p align='justify'>Linear regression is a supervised learining algorithm used when target variable continues real number. It establishes a relationship between an independent variable x & a dependent variable y using a best fit line</p>", unsafe_allow_html=True)

    article_link, source_code_link = st.columns([1,1.13])

    link_to_article = 'https://towardsdatascience.com/linear-regression-5100fe32993a'
    link_to_source_code = 'https://github.com/thisisashwinraj/Machine-Learning-Recipes/blob/main/regression/linear_regression.py'
    
    with article_link:
        st.markdown("<b> :page_facing_up: Medium Blog:</b> <a href=" + link_to_article + ">Simple Linear Regression</a>", unsafe_allow_html=True)
    
    with source_code_link:
        st.markdown("<b> :male-technologist: GitHub:</b> <a href=" + link_to_source_code + ">thisisashwinraj/Machine-Learning-Recipes</a>", unsafe_allow_html=True)

    simple_linear_regression_metrics_expander = st.expander("Explore Model Performance")

    with simple_linear_regression_metrics_expander:
        error_refinement_collective, association_equilibrium_ensemble = st.columns(2)

        mse, r2score, model_intercept, line_coefficient = LinearRegressionModel.model_evaluation("datasets/diamonds.csv")

        with error_refinement_collective:
            st.markdown("<b>:u7121: Mean Square Error: </b> {:.3f}".format(mse), unsafe_allow_html=True)
            st.markdown("<b>:u6709: Regression Intercept: </b> {:.4f}".format(model_intercept.item()), unsafe_allow_html=True)

        with association_equilibrium_ensemble:
            st.markdown("<b>:u6708: Regression R2 Score: </b> {:.4f}".format(r2score), unsafe_allow_html=True)
            st.markdown("<b>:u55b6: LR Line Coefficient: </b> {:.3f}".format(line_coefficient[0].item()), unsafe_allow_html=True)

    user_input = st.number_input('Enter the Carats:', min_value=0.2, max_value=5.0, value=1.0, step=0.1, help='Calculate the price of a diamond based on its carat value')

    model_inferencing, model_recalibration = st.columns([1,2.6])
    linear_regression_model = LinearRegressionModel()

    with model_inferencing:
        infer_simple_linear_regression_model_button = st.button("Predict Diamond's Price")

    with model_recalibration:
        recalibrate_simple_linear_regression_model_button = st.button("Recalibrate")

    if infer_simple_linear_regression_model_button:
        predicted_price = linear_regression_model.inference_from_model(user_input)
        linear_regression_model_prediction = st.success(f":gem: Predicted Price for a {user_input:.2f} carat diamond is ${predicted_price.item():.2f}")

    if recalibrate_simple_linear_regression_model_button:
        linear_regression_model_recalibration = st.spinner("Recalibrating the Simple Linear Regression Model")

        with linear_regression_model_recalibration:
            try:
                linear_regression_model.train_regression_model("datasets/diamonds.csv")

                model_filename = 'inference/linear_regression_model.pkl'
                model = joblib.load(model_filename)
                time.sleep(2)
                
                recalibration_failed_alert = False

            except:
                recalibration_failed_alert = True

        if not recalibration_failed_alert:
            linear_regression_model_recalibration_success = st.info("Linear Regression Model Recalibrated Succesfully", icon='ℹ️')
            time.sleep(2)

            linear_regression_model_recalibration_success.empty()

        else:
            linear_regression_model_recalibration_failed = st.warning("Linear Regression Model Could'nt be Recalibrated")
            time.sleep(2)

            linear_regression_model_recalibration_failed.empty()


def multivariate_regression():
    #To be added shortly
    pass

def decision_tree_regression():
    #To be added shortly
    pass

def random_forest_regression():
    #To be added shortly
    pass

def svm_based_regression():
    #To be added shortly
    pass
