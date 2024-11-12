import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from utils.data_analyzer import (get_basic_info, analyze_data_types, analyze_duplicates, 
                               get_numerical_stats, analyze_outliers, analyze_trends_and_seasonality, detect_anomalies)
from utils.data_visualizer import (create_histogram, create_box_plot, create_scatter_plot,
                                plot_correlation_matrix, plot_missing_values, plot_outliers)
from utils.data_processor import (change_column_type, handle_missing_values, 
                               remove_duplicates, export_data)
from utils.database import (delete_dataframe, get_table_info)
from utils.report_generator import generate_data_report
from pathlib import Path

def show_overview_tab(df):
    st.header("Обзор")
    get_basic_info(df)
    st.dataframe(df.head())
    analyze_data_types(df)

def show_analysis_tab(df):
    st.header("Анализ")
    analyze_duplicates(df)
    get_numerical_stats(df)
    plot_missing_values(df)

    # --- Анализ трендов и сезонности ---
    st.subheader("Анализ трендов и сезонности")
    
    # Определение столбцов с датами
    date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
    # Если нет датафрейм столбцов, проверяем object столбцы
    if not date_columns:
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                continue
                
    # Определение числовых столбцов
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not date_columns:
        st.warning("В датасете не найдены столбцы с датами")
        return
        
    if not numeric_columns:
        st.warning("В датасете не найдены числовые столбцы")
        return
        
    date_column = st.selectbox(
        "Выберите столбец даты",
        date_columns,
        key='date_column_select'
    )
    
    value_column = st.selectbox(
        "Выберите столбец значений",
        numeric_columns,
        key='value_column_select'
    )
    
    if date_column and value_column:
        analyze_trends_and_seasonality(df, date_column, value_column)

    # --- Обнаружение аномалий ---
    st.subheader("Обнаружение аномалий")
    anomaly_column = st.selectbox(
        "Выберите столбец для обнаружения аномалий", 
        numeric_columns,
        key='anomaly_column_select'
    )
    if anomaly_column:
        detect_anomalies(df, anomaly_column)

def show_visualization_tab(df):
    st.header("Визуализация")
    viz_type = st.selectbox(
        "Выберите тип визуализации", 
        ["Гистограмма", "Box Plot", "Scatter Plot", "Корреляционная матрица"],
        key="viz_type"
    )
    
    if viz_type in ["Гистограмма", "Box Plot"]:
        column = st.selectbox("Выберите столбец", df.select_dtypes(include=[np.number]).columns)
        if viz_type == "Гистограмма":
            create_histogram(df, column)
        else:
            create_box_plot(df, column)
    elif viz_type == "Scatter Plot":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        x_column = st.selectbox("Выберите X", numerical_cols)
        y_column = st.selectbox("Выберите Y", numerical_cols)
        create_scatter_plot(df, x_column, y_column)
    else:
        plot_correlation_matrix(df)

def show_preprocessing_tab(df):
    st.header("Предобработка")
    process_type = st.selectbox(
        "Выберите тип обработки", 
        ["Изменение типов данных", "Обработка пропусков", "Удаление дубликатов"]
    )
    
    if process_type == "Изменение типов данных":
        column = st.selectbox("Выберите столбец", df.columns)
        current_type = str(df[column].dtype)
        
        # Определение возможных типов данных
        type_options = {
            'Целое число (int)': 'int64',
            'Число с плавающей точкой (float)': 'float64',
            'Строка (string)': 'str',
            'Дата (datetime)': 'datetime64',
            'Булево значение (bool)': 'bool'
        }
        
        new_type = st.selectbox(
            f"Текущий тип: {current_type}. Выберите новый тип:", 
            list(type_options.keys())
        )
        
        if st.button("Применить изменение типа"):
            selected_type = type_options[new_type]
            df, success = change_column_type(df, column, selected_type)
            
            if success:
                st.session_state['df'] = df
                st.success(f"✅ Тип столбца {column} успешно изменен на {new_type}")
                st.dataframe(df.head())  # Отобразить обновленный DataFrame

    elif process_type == "Обработка пропусков":
        column = st.selectbox("Выберите столбец", df.columns)
        method = st.selectbox(
            "Выберите метод обработки",
            ["Удалить строки с пропусками", "Заполнить значением", "Заполнить средним", "Заполнить медианой"]
        )
        value = None
        if method == "Заполнить значением":
            value = st.text_input("Введите значение для заполнения")
        if st.button("Применить"):
            method_mapping = {
                "Удалить строки с пропусками": "drop",
                "Заполнить значением": "fill_value",
                "Заполнить средним": "fill_mean",
                "Заполнить медианой": "fill_median"
            }
            selected_method = method_mapping[method]
            df, success = handle_missing_values(df, column, selected_method, value)
            if success:
                st.session_state['df'] = df
                st.success("✅ Обработка пропусков успешно выполнена")
                st.dataframe(df.head())  # Отобразить обновленный DataFrame
                
    elif process_type == "Удаление дубликатов":
        subset = st.multiselect("Выберите столбцы для проверки дубликатов (необязательно)", df.columns)
        keep_option = st.selectbox("Какие дубликаты сохранить?", ["Первый", "Последний", "Удалить все"])
        keep_mapping = {
            "Первый": "first",
            "Последний": "last",
            "Удалить все": False
        }
        if st.button("Удалить дубликаты"):
            keep = keep_mapping[keep_option]
            df, success = remove_duplicates(df, subset=subset if subset else None, keep=keep)
            if success:
                st.session_state['df'] = df
                st.success("✅ Дубликаты успешно удалены")
                st.dataframe(df.head())  # Отобразить обновленный DataFrame

def show_export_tab(df):
    st.header("Экспорт")
    format_type = st.selectbox("Выберите формат экспорта", ['csv', 'excel'])
    if st.button("Экспортировать"):
        result = export_data(df, format_type)
        if result:
            data, mime_type, file_ext = result
            st.download_button(
                "⬇️ Скачать файл",
                data,
                f"export.{file_ext}",
                mime_type
            )

def show_database_tab(df):
    st.header("База данных")
    table_info = get_table_info()
    if table_info:
        for key, value in table_info.items():
            st.write(f"**{key}:** {value}")
    if st.button("❌ Очистить базу данных"):
        if delete_dataframe():
            st.session_state.pop('df', None)
            st.success("✅ База данных успешно очищена")
            st.rerun()

def show_reports_tab(df):
    st.header("Отчеты")
    report_sections = st.multiselect(
        "Выберите разделы для отчета",
        ["Базовая информация", "Типы данных", "Статистика", 
         "Пропущенные значения", "Дубликаты"],
        default=["Базовая информация", "Типы данных"]
    )
    
    if st.button("📄 Сгенерировать отчет"):
        if report_sections:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result = generate_data_report(
                    df=df,
                    sections=report_sections,
                    fname=f'report_{timestamp}.xlsx'
                )
                if result:
                    st.success("✅ Отчет успешно создан")
                    st.download_button(
                        "⬇️ Скачать отчет",
                        open(result, 'rb').read(),
                        os.path.basename(result),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            except Exception as e:
                st.error(f"Ошибка при создании отчета: {str(e)}")
