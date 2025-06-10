import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Rakuten: multi-modal product classification project")
st.sidebar.title("Table of contents")
pages=["Introduction", "Data Exploration", "Data Preprocessing", "Modeling", "DEMO app"]
page=st.sidebar.radio("Go to", pages)

original_dataset = pd.read_csv('../data/X_train.csv', index_col=0)
y_ori = pd.read_csv('../data/Y_train.csv', index_col=0)
original_dataset = pd.concat([original_dataset, y_ori], axis=1)

if page == pages[0]:
    st.write("### Introduction of the project")

if page == pages[1]:
    st.write("### Presentation of data")
    st.write("We first explore the data available for our project, here is an short overview of first five rows of data:")
    st.dataframe(original_dataset.head())

if page == pages[2]:
    st.write("### Preprocessing of data")

if page == pages[3]:
    st.write("### Modeling")

if page == pages[4]:
    st.write("### DEMO app: try out with your own input")