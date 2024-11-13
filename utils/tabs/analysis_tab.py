import streamlit as st
from utils.data_analyzer import analyze_duplicates, get_numerical_stats
from utils.data_visualizer import plot_missing_values

def show_analysis_tab(df):
    st.header("Анализ")
    analyze_duplicates(df)
    get_numerical_stats(df)
    plot_missing_values(df)
    # ...остальной код...
