import numpy as np
import pandas as pd
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataWrangling:

    def impute_missing_values(self, data):
        missing_values = data.isnull().sum()

        if missing_values.any():
            data = data.fillna(data.mean())

        return data

    def remove_duplicates(self, data):        
        data = data.drop_duplicates()
        return data

    def remove_outliers(self, data):
        numeric_data = data.select_dtypes(include=['number'])

        q1 = numeric_data.quantile(0.25)
        q3 = numeric_data.quantile(0.75)

        iqr = q3 - q1

        numeric_data = numeric_data[~((numeric_data < (q1 - 1.5 * iqr)) | (numeric_data > (q3 + 1.5 * iqr))).any(axis=1)]
        cleaned_data = pd.concat([numeric_data, data.select_dtypes(exclude=['number'])], axis=1)

        return cleaned_data


class FeatureEngineering:

    def drop_unnecessary_features(self, data):
        data = data.drop("Unnamed: 0", axis=1)

        X = data['carat']
        y = data['price']

        return X, y

    def scale_and_normalize(self, data):
        scaler = StandardScaler()

        data = scaler.fit_transform(data)
        return data


class LinearRegressionModel:

    def inference_from_model(self, input_data):
        with open("inference/linear_regression_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        predicted_price = model.predict([[input_data]])[0]
        return predicted_price
    
    def train_regression_model(self, filename):
        data = pd.read_csv(filename)

        try:
            data_wrangling = DataWrangling()
            data = data_wrangling.impute_missing_values(data)

            data = data_wrangling.remove_duplicates(data)
            data = data_wrangling.remove_outliers(data)

        except Exception as e:
            print("Model training failed during data preprocessing")
        
        try:
            X, y = data['carat'], data['price']
            X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)

            X_train_array = np.array(X_train)
            X_train = X_train_array[:,np.newaxis]

            X_test_array = np.array(X_test)
            X_test = X_test_array[:,np.newaxis]

        except Exception as e:
            print("Model training failed during feature engineering")

        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        except:
            print("Model training failed during data scaling")

        try:
            imputer = SimpleImputer(strategy='mean')

            X_train_imputed = imputer.fit_transform(X_train_scaled)
            y_train_imputed = imputer.fit_transform(y_train.values.reshape(-1, 1))

            linear_regression_model = LinearRegression()
            linear_regression_model.fit(X_train_imputed,y_train_imputed)

        except Exception as e:
            print("Model training failed during build")

        with open("inference/linear_regression_model.pkl", "wb") as model_file:
            pickle.dump(linear_regression_model, model_file)
        
        return 0

    def model_evaluation(filename):
        data = pd.read_csv(filename)
        X, y = data['carat'], data['price']
        
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=100)

        X_train_array = np.array(X_train)
        X_train = X_train_array[:,np.newaxis]

        X_test_array = np.array(X_test)
        X_test = X_test_array[:,np.newaxis]

        with open("inference/linear_regression_model.pkl", "rb") as model_file:
            linear_regression_model = pickle.load(model_file)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_pred = linear_regression_model.predict(X_test_scaled)

        mse = mean_squared_error(y_test,y_pred)
        r2score = r2_score(y_test,y_pred)

        model_intercept = linear_regression_model.intercept_
        line_coefficient = linear_regression_model.coef_

        return mse, r2score, model_intercept, line_coefficient
    
    def plot_static_regression_line(data_filename, feature_to_visualize):
        data = pd.read_csv(data_filename)
        
        with open("inference/linear_regression_model.pkl", "rb") as model_file:
            linear_regression_model = pickle.load(model_file)

        scaler = StandardScaler()
        feature_data = data[[feature_to_visualize]]

        scaled_feature = scaler.fit_transform(feature_data)
        
        predicted_prices = linear_regression_model.predict(scaled_feature)

        plt.figure(figsize=(12, 6))

        plt.scatter(data[feature_to_visualize], data["price"], alpha=0.3, label="Actual Values")
        plt.plot(data[feature_to_visualize], predicted_prices, color="red", linewidth=3, label="Regression Line")

        plt.xlabel(feature_to_visualize)
        plt.ylabel("Price")

        plt.xlim(0, 3)
        plt.ylim(0, 23000)

        plt.title("Scatter Plot with Regression Line")
        plt.legend()

        return plt
    
    def plot_interactive_regression_line(data_filename, feature_to_visualize):
        data = pd.read_csv(data_filename)
        with open("inference/linear_regression_model.pkl", "rb") as model_file:
            linear_regression_model = pickle.load(model_file)

        scaler = StandardScaler()
        feature_data = data[['carat']]

        feature_data = scaler.fit_transform(feature_data)
        scaled_feature = scaler.transform(feature_data)
        
        predicted_prices = linear_regression_model.predict(scaled_feature)

        fig = px.scatter(data, x=data[feature_to_visualize], y=data["price"])
        fig.update_traces(marker=dict(size=12, opacity=0.6),
                        selector=dict(mode='markers'))
        fig.update_layout(showlegend=True)

        fig.update_layout(
            xaxis=dict(fixedrange=True),
            yaxis=dict(fixedrange=True),
            width=300,
            height=300
        )
        fig.add_traces(px.scatter(data, x=data[feature_to_visualize], y=data["price"], trendline="ols").data)
        
        return fig


if __name__ == '__main__':

    linear_regression_model = LinearRegressionModel()
    linear_regression_model.train_regression_model("datasets/diamonds.csv")

    input_carat = float(input("Enter the diamond's carat value: "))
    predicted_price = linear_regression_model.inference_from_model(input_carat)

    print(f"The predicted price of {input_carat} carat diamond is ${predicted_price.item():.2f}")
