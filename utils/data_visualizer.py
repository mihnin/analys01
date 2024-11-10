import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def create_histogram(df, column):
    """
    Создание гистограммы
    """
    fig = px.histogram(df, x=column, title=f'Гистограмма: {column}')
    st.plotly_chart(fig)

def create_box_plot(df, column):
    """
    Создание box plot
    """
    fig = px.box(df, y=column, title=f'Box Plot: {column}')
    st.plotly_chart(fig)

def create_scatter_plot(df, x_column, y_column):
    """
    Создание scatter plot
    """
    fig = px.scatter(df, x=x_column, y=y_column, 
                    title=f'Scatter Plot: {x_column} vs {y_column}')
    st.plotly_chart(fig)

def plot_correlation_matrix(df):
    """
    Построение корреляционной матрицы
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Корреляция"),
                       title="Корреляционная матрица")
        st.plotly_chart(fig)
    else:
        st.info("Недостаточно числовых столбцов для построения корреляционной матрицы")

def plot_missing_values(df):
    """
    Визуализация пропущенных значений
    """
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(x=missing_data.index, 
                    y=missing_data.values,
                    title="Количество пропущенных значений по столбцам")
        st.plotly_chart(fig)
    else:
        st.info("В датасете нет пропущенных значений")
