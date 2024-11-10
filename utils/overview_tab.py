import streamlit as st
from utils.data_analyzer import get_basic_info, analyze_data_types

def show_overview_tab(df):
    st.header("Обзор")
    get_basic_info(df)
    st.dataframe(df.head())
    analyze_data_types(df)
