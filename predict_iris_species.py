# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split as tts

# Loading the dataset.
iris_df = pd.read_csv("iris-species.csv")

# Adding a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
# Creating the numeric target column 'Label' to 'iris_df' using the 'map()' function.
iris_df['Label'] = iris_df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

# Creating a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

# Creating features and target DataFrames.
f = iris_df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
t = iris_df['Label']

# Splitting the data into training and testing sets.
xtrain, xtest, ytrain, ytest = tts(f, t, test_size = 0.33, random_state = 42)

@st.cache()

def prediction(SepalLength, SepalWidth, PetalLength, PetalWidth, pm):
  species = pm.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
  species = species[0]
  if species == 0:
    return "Iris Setosa"
  elif species == 1:
    return "Iris Virginica"
  else:
    return "Iris Versicolor"

mt = [
  "Support Vector Machine: Very accurate, but slow",
  "Logistic Regression: Quite accurate, but a bit slower",
  "Linear Regression: Balanced",
  "Random Forest Classifier: Very fast, but not as accurate"
]
# Add title widget
st.title("Iris Flower Species Prediction App")  

# Creating the SVC model and storing the accuracy score in a variable 'score'.


classifier = st.sidebar.selectbox("", ("Support Vector Machine", "Logistic Regression", "Linear Regression", "Random Forest Classifier"))

st.sidebar.title("Select Classifier:", classifier)

if classifier == 'Support Vector Machine':
  from sklearn.svm import SVC as svc
  pm = svc(kernel='linear').fit(xtrain, ytrain)
  score = pm.score(xtrain, ytrain)

elif classifier =='Linear Regression':
  from sklearn.linear_model import LinearRegression as lreg
  pm = lreg().fit(xtrain, ytrain)
  score = pm.score(xtrain, ytrain)

elif classifier =='Logistic Regression':
  from sklearn.linear_model import LogisticRegression as lgreg
  pm = lgreg().fit(xtrain, ytrain)
  score = pm.score(xtrain, ytrain)

else:
  from sklearn.ensemble import RandomForestClassifier as rfc
  pm = rfc().fit(xtrain, ytrain)
  score = pm.score(xtrain, ytrain)



st.sidebar.title("Prediction Model Info:")

for i in mt:
  st.sidebar.write(i)

st.sidebar.title("Iris Measurements:")

# Add 4 sliders and store the value returned by them in 4 separate variables.

sl = st.sidebar.slider("Sepal Length:", 0.0, 10.0)
sw = st.sidebar.slider("Sepal Width:", 0.0, 10.0)
pl = st.sidebar.slider("Petal Length:", 0.0, 10.0)
pw = st.sidebar.slider("Petal Width:", 0.0, 10.0)


# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.sidebar.button("Predict"):
	species = prediction(sl, sw, pl, pw, pm)
	st.write(f"Predicted Iris Species: {species}.")
	st.write(f"This model's accuracy rate is {score*100}%.")