import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Preprocessing modules
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow import keras

from joblib import dump, load
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 123
class LinearBlock(keras.layers.Layer):
    def __init__(self, neurons, kernel_initializer=tf.keras.initializers.HeNormal(seed=seed), activation="relu", dropout = 0.0):
        super(LinearBlock, self).__init__()
        self.linear = Dense(neurons, kernel_regularizer = tf.keras.regularizers.L2(0.01),
                            kernel_initializer=kernel_initializer)
        self.bn = BatchNormalization()
        self.relu = Activation(activation)
        self.dropout = Dropout(dropout)
    def call(self, inputs):
        x = self.linear(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)

optimizer = Adam(0.001)
def make_model():
    inputs = keras.Input(shape=(16))
    x = LinearBlock(16, )(inputs)
    # x = LinearBlock(16, )(x)
    x = tf.keras.layers.Add()([x, inputs])
    x = LinearBlock(8, dropout = 0.2)(x)
    outputs = Dense(1, activation="sigmoid", kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(x)
    model2 = tf.keras.models.Model(inputs, outputs)
    model2.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model2

normalizer = tf.keras.models.load_model(filepath = 'normalizer.tf')
model = make_model()
filepath = ("./checkpoint/")
model.load_weights(filepath)
ohe = load('onehot_encoder.joblib')
normalizer = tf.keras.models.load_model(filepath = 'normalizer.tf')
sorted_idx = np.load("./sorted_idx.npy")

# Sidebar widget
st.sidebar.header('Menu')
# loading our model
# model = joblib.load("./model.pkl")  
new_data_test = pd.read_csv('train.csv', )

# def main():
#     page = st.sidebar.selectbox(
#         "Select a page", ["Prediction"])

#     if page == "Prediction":
#         model_predict()

@st.cache()
def load_data():
    data = pd.read_csv('./WA_Fn-UseC_-Telco-Customer-Churn.csv')
    return data


df = load_data()
df = df.drop(columns=["Churn", "customerID"])

categorical = df.select_dtypes(["object"]).columns.to_list()
numerical = [i for i in df if i not in categorical]

data = dict.fromkeys(df.columns)
for col in categorical:
    data[col] = list(df[col].unique())
for col in numerical:
    data[col] = [df[col].min(), df[col].max()]

new_data = data.copy()

# def model_predict():
st.title("Prediction")
st.write("### Field this form to predict if the customer gonna churn or not!")

new_data["gender"] = st.radio(
    "Gender", data["gender"])

binary_option = {1: "Yes", 0: "No"}
new_data["SeniorCitizen"] = st.radio(
    "Senior Citizen", [1, 0], format_func=(lambda x: binary_option.get(x)))

new_data["Partner"] = st.radio(
    "Partner", ["Yes", "No"],)

new_data["Dependents"] = st.radio(
    "Dependents", ["Yes", "No"],)

new_data["tenure"] = st.number_input(
    label="tenure", min_value=1, max_value=None, step=1, ) 

new_data["PhoneService"] = st.radio(
    "PhoneService", ["Yes", "No"],)

new_data["MultipleLines"] = st.radio(
    "MultipleLines", data["MultipleLines"])

new_data["InternetService"] = st.radio(
    "InternetService", data["InternetService"])

new_data["OnlineSecurity"] = st.radio(
    "OnlineSecurity", data["OnlineSecurity"])

new_data["OnlineBackup"] = st.radio(
    "OnlineBackup", data["OnlineBackup"])

new_data["DeviceProtection"] = st.radio(
    "DeviceProtection", data["DeviceProtection"])

new_data["TechSupport"] = st.radio(
    "TechSupport", data["TechSupport"])

new_data["StreamingTV"] = st.radio(
    "StreamingTV", data["StreamingTV"])

new_data["StreamingMovies"] = st.radio(
    "StreamingMovies", data["StreamingMovies"])

new_data["Contract"] = st.radio(
    "Contract", data["Contract"])

new_data["PaperlessBilling"] = st.radio(
    "PaperlessBilling", data["PaperlessBilling"])

new_data["PaymentMethod"] = st.radio(
    "PaymentMethod", data["PaymentMethod"])

new_data["MonthlyCharges"] = st.number_input(
    label="MonthlyCharges", min_value=1, max_value=None, step=1, ) 

new_data["TotalCharges"] = st.number_input(
    label="TotalCharges", min_value=1, max_value=None, step=1, )


submit_button = st.button("Predict")

new_data = pd.DataFrame([new_data])

def predict_classes(model, data):
    return np.argmax(model.predict(data), axis=-1)[0]

def predict(new_data):
    categorical = new_data.select_dtypes(["object"]).columns.to_list()
    numerical = [i for i in new_data if i not in categorical]
    categorical_inf = ohe.transform(new_data[categorical].astype(str)).toarray()
    categorical_inf = pd.DataFrame(categorical_inf, columns=ohe.get_feature_names_out())

    numerical_inf = new_data[numerical].copy()
    tf.convert_to_tensor(numerical_inf)
    numerical_inf = normalizer.predict(numerical_inf)
    numerical_inf = pd.DataFrame(numerical_inf, columns=numerical)

    X_inf = pd.concat((categorical_inf, numerical_inf), axis=1)

    feature_num = 16
    cluster_inf = X_inf[X_inf.columns[sorted_idx[-feature_num:]]]
    tf.convert_to_tensor(cluster_inf)

    return ("Not Leave" if predict_classes(model, cluster_inf) == 0 else "Leave")

if submit_button:
    result = predict(new_data)

    # updated_res = result[0] #gridcv.predict(new_data)[0]

    st.success(
        'This customer will %s this month' %result)


# main()
