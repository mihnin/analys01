import pandas as pd
import numpy as np
import streamlit as st

def get_basic_info(df):
    """
    Получение базовой информации о датасете
    """
    st.subheader("Обзор данных")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Количество строк", df.shape[0])
    with col2:
        st.metric("Количество столбцов", df.shape[1])
    with col3:
        st.metric("Размер данных (MB)", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))

def analyze_data_types(df):
    """
    Анализ типов данных
    """
    st.subheader("Типы данных")
    dtypes_df = pd.DataFrame({
        'Столбец': df.dtypes.index,
        'Тип данных': df.dtypes.values,
        'Количество null': df.isnull().sum().values,
        'Процент null': (df.isnull().sum().values / len(df) * 100).round(2)
    })
    st.dataframe(dtypes_df)

def analyze_duplicates(df):
    """
    Анализ дубликатов
    """
    st.subheader("Анализ дубликатов")
    
    duplicates = df.duplicated().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Количество дубликатов", duplicates)
    with col2:
        st.metric("Процент дубликатов", round(duplicates / len(df) * 100, 2))

def get_numerical_stats(df):
    """
    Получение статистики по числовым данным
    """
    st.subheader("Статистика числовых данных")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        stats_df = df[numerical_cols].describe()
        st.dataframe(stats_df)
    else:
        st.info("В датасете нет числовых столбцов")
