import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Preprocessing modules
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Machine Learning metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.base import TransformerMixin, BaseEstimator

# Machine Learning model
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

# Deployment purposes
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer


class mechanicTransform(BaseEstimator,TransformerMixin): 
    def __init__(self,):
        self.mlb = MultiLabelBinarizer()

    def fit_transform(self, df, y=None, **fit_params):
        return pd.DataFrame(self.mlb.fit_transform(df['mechanic'].str.split(', ')),columns=self.mlb.classes_, index=df.index)

    def fit(self, df, y=None, **fit_params): 
        self.mlb.fit(df['mechanic'].str.split(', '))
        return self

    def transform(self, df, y=None, **fit_params): 
        return pd.DataFrame(self.mlb.transform(df['mechanic'].str.split(', ')),columns=self.mlb.classes_, index=df.index)

class categoryTransform(BaseEstimator,TransformerMixin): 
    def __init__(self,):
        self.mlb = MultiLabelBinarizer()

    def func1(self, df):
        df = pd.DataFrame(df, columns=(f"category{i+1}" for i in range(12)))
        for i in range(12):
            df[f"category{i+1}"] = df[f"category{i+1}"].str.lstrip()
        df_category = df[(f"category{i+1}" for i in range(12))].apply(lambda x: ",".join(x), axis=1).str.split(',')
        return df_category

    def fit_transform(self, df, y=None, **fit_params):
        df_category = self.func1(df)
        return self.mlb.fit_transform(df_category)

    def fit(self, df, y=None, **fit_params): 
        df_category = self.func1(df)
        self.mlb.fit(df_category)
        return self
    
    def transform(self, df, y=None, **fit_params): 
        df_category = self.func1(df)
        return self.mlb.transform(df_category)

class SkewTransform(TransformerMixin): 
  def fit(self, X, y=None, **fit_params): 
    return self 
  def transform(self, df, y=None, ): 
    return np.log10(1+df)

    
# Sidebar widget
st.sidebar.header('Menu')
# loading our model
model = joblib.load("./model.pkl")  
new_data_test = pd.read_csv('train.csv', )

# def main():
#     page = st.sidebar.selectbox(
#         "Select a page", ["Prediction"])

#     if page == "Prediction":
#         model_predict()


@st.cache()
def load_data():
    data = pd.read_csv('train.csv')
    return data


df = load_data()
df = df.drop(columns=["game_id", "names", "designer", "geek_rating", "min_time", "max_time", "age"])


cat_col = df.select_dtypes(["object"]).columns.to_list()
df[cat_col] = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="None").fit_transform(df[cat_col])


mlb = MultiLabelBinarizer()
for i in range(12):
    df[f"category{i+1}"] = df[f"category{i+1}"].str.lstrip()
df_category = df[(f"category{i+1}" for i in range(12))].apply(lambda x: ",".join(x), axis=1).str.split(',')
df_category = pd.DataFrame(mlb.fit_transform(df_category),columns=mlb.classes_, index=df.index)

mlb2 = MultiLabelBinarizer()
df_mechanic = pd.DataFrame(mlb2.fit_transform(df['mechanic'].str.split(', ')),columns=mlb2.classes_, index=df.index)
df_mechanic_new = df_mechanic[0:0]
df_category_new = df_category[0:0]

# def model_predict():
st.title("Prediction")
st.write("### Field this form to predict the rating of your board game !")
min_players = df.min_players
max_players = df.max_players
avg_time = list(df.avg_time.unique())
year = list(df.year.unique())
num_votes = list(df.num_votes.unique())
mechanic = list(df_mechanic.columns)
owned = list(df.owned.unique())
category = list(df_category.columns)

min_players_in = st.slider(
    "Number of minimum players", 0, 8)
max_players_in = st.slider(
    "Number of maximum players", 0, 200)
avg_time_in = st.number_input(
    label="Number of average time", min_value=0, max_value=None, step=1, ) 
year_in = st.number_input(
    label="In which year the board game created?", min_value=0, max_value=None, step=1, ) 
num_votes_in = st.number_input(
    label="Number of votes", min_value=0, max_value=None, step=1, ) #format="%.2f"
owned_in = st.number_input(
    label="Number of people owned the board games", min_value=0, max_value=None, step=1, )

st.subheader('Select mechanic:')

mechanic_in = np.zeros(len(mechanic))
category_in = np.zeros(len(category))
for i, x in enumerate(mechanic):
    df_mechanic_new[x] = st.checkbox(x, value=False, key=f"mech_{i}")

st.subheader('Select category:')
for i, x in enumerate(category):
    df_category_new[x] = st.checkbox(x, value=False, key=f"cate_{i}")


submit_button = st.button("Predict")

#     ['min_players', 'max_players', 'avg_time', 'year', 'num_votes', 'mechanic', 'owned', 
# 'category1', 'category2', 'category3', 'category4', 'category5', 'category6', 'category7', 
# 'category8', 'category9', 'category10', 'category11', 'category12']
flag1 = True
mechanic_in = ""
for col in df_mechanic_new.columns:
    if(list(df_mechanic_new[col])):
        if(flag1 == True):
            mechanic_in = f", {col}"
        else:
            mechanic_in += f"{col}"
            flag1 = False

n=0
flag2 = 0
category = np.zeros(12)
for col in df_category_new.columns:
    if(list(df_category_new[col])):
        if(flag2<12):
            category[n] = str(col)
            flag2 += 1
category = np.where(category == 0, "None", category)
data = {
    'min_players': [min_players_in], 'max_players': [max_players_in], 'avg_time': [avg_time_in], 'year': [year_in], 'num_votes': [num_votes_in],
    'owned': [owned_in], 'mechanic': [mechanic_in], 'category1': [category[0]], 'category2': [category[1]], 'category3': [category[2]], 
    'category4': [category[3]], 'category5': [category[4]], 'category6': [category[5]], 'category7': [category[6]], 'category8': [category[7]],
    'category9': [category[8]], 'category10': [category[9]], 'category11': [category[10]], 'category12': [category[11]]
}

new_data = pd.DataFrame(data=data)

if submit_button:
    result = model.predict(new_data)

    updated_res = result[0] #gridcv.predict(new_data)[0]

    st.success(
        'The Rating for this Board Games is {:.2f}'.format(updated_res))


# main()
