import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.title("IRIS flower prediction app")
st.sidebar.title("input parameters")

def load():
    sl=st.sidebar.slider('Sepal length',4.3,7.9,5.4)
    sw=st.sidebar.slider('Sepal width',2.0,4.4,3.4)
    pl=st.sidebar.slider('Petal length',1.0,6.9,1.3)
    pw=st.sidebar.slider('Petal width',0.1,2.5,0.2)
    data={
        'sepal_length':sl,
        'sepal_width':sw,
        'petal_length':pl,
        'petal_width':pw
        }
    festures=pd.DataFrame(data,index=[0])
    return festures


df=load()
st.subheader('User input')
st.write(df)
iris=datasets.load_iris()
x=iris.data
y=iris.target

st.subheader('Labels')
st.write(iris.target_names)

clf=RandomForestClassifier()
clf.fit(x,y)
pre=clf.predict(df)
pre_pro=clf.predict_proba(df)

st.subheader('Prediction')
st.write(iris.target_names[pre])
st.subheader('prediction probability')
st.write(pre_pro)