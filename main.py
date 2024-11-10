import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import get_file_uploader
from utils.data_analyzer import (get_basic_info, analyze_data_types, 
                               analyze_duplicates, get_numerical_stats,
                               analyze_outliers)
from utils.data_visualizer import (create_histogram, create_box_plot, 
                                create_scatter_plot, plot_correlation_matrix,
                                plot_missing_values, plot_outliers)
from utils.data_processor import (change_column_type, handle_missing_values,
                               remove_duplicates, export_data)
from utils.predictor import (prepare_data, train_model, evaluate_model,
                          plot_feature_importance, plot_predictions)

def load_test_data():
    """
    Загрузка тестового набора данных
    """
    try:
        return pd.read_csv('test_data.csv')
    except Exception as e:
        st.error(f"Ошибка при загрузке тестовых данных: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Анализ данных",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Комплексный анализ данных")
    
    # Секция загрузки данных
    st.subheader("Загрузка данных")
    
    # Кнопка загрузки тестовых данных
    if st.button("📥 Загрузить тестовые данные"):
        test_df = load_test_data()
        if test_df is not None:
            st.session_state['df'] = test_df
            st.success("Тестовые данные успешно загружены!")
            
    # Стандартный загрузчик файлов
    uploaded_df = get_file_uploader()
    if uploaded_df is not None:
        st.session_state['df'] = uploaded_df
    
    # Работа с загруженными данными
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Создание вкладок
        tabs = st.tabs(["Обзор", "Анализ", "Визуализация", "Прогнозирование", "Предобработка", "Экспорт"])
        
        # Вкладка обзора
        with tabs[0]:
            get_basic_info(df)
            st.dataframe(df.head())
            analyze_data_types(df)
        
        # Вкладка анализа
        with tabs[1]:
            analyze_duplicates(df)
            get_numerical_stats(df)
            plot_missing_values(df)
            
            # Анализ выбросов
            st.subheader("Анализ выбросов")
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                selected_column = st.selectbox(
                    "Выберите столбец для анализа выбросов",
                    numerical_cols,
                    key="outliers_column"
                )
                lower_bound, upper_bound = analyze_outliers(df, selected_column)
                plot_outliers(df, selected_column, lower_bound, upper_bound)
            else:
                st.info("В датасете нет числовых столбцов для анализа выбросов")
        
        # Вкладка визуализации
        with tabs[2]:
            st.subheader("Визуализация данных")
            
            viz_type = st.selectbox("Выберите тип визуализации", 
                                 ["Гистограмма", "Box Plot", "Scatter Plot", "Корреляционная матрица"],
                                 key="viz_type")
            
            if viz_type in ["Гистограмма", "Box Plot"]:
                column = st.selectbox("Выберите столбец", 
                                   df.select_dtypes(include=[np.number]).columns,
                                   key="viz_column")
                if viz_type == "Гистограмма":
                    create_histogram(df, column)
                else:
                    create_box_plot(df, column)
                    
            elif viz_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_column = st.selectbox("Выберите X", 
                                         df.select_dtypes(include=[np.number]).columns,
                                         key="scatter_x")
                with col2:
                    y_column = st.selectbox("Выберите Y", 
                                         df.select_dtypes(include=[np.number]).columns,
                                         key="scatter_y")
                create_scatter_plot(df, x_column, y_column)
                
            else:
                plot_correlation_matrix(df)
        
        # Вкладка прогнозирования
        with tabs[3]:
            st.subheader("Прогнозирование")
            
            # Выбор типа задачи
            task_type = st.selectbox(
                "Выберите тип задачи",
                ["Регрессия", "Классификация"],
                key="task_type"
            )
            task_type = task_type.lower()
            
            # Выбор целевой переменной
            if task_type == "регрессия":
                target_columns = df.select_dtypes(include=[np.number]).columns
            else:
                target_columns = df.select_dtypes(include=['object', 'category']).columns
                
            target = st.selectbox(
                "Выберите целевую переменную",
                target_columns,
                key="target"
            )
            
            # Выбор признаков
            feature_columns = st.multiselect(
                "Выберите признаки для прогнозирования",
                [col for col in df.columns if col != target],
                key="features"
            )
            
            if st.button("Обучить модель") and len(feature_columns) > 0:
                with st.spinner("Обучение модели..."):
                    # Подготовка данных
                    X_train, X_test, y_train, y_test, scaler = prepare_data(
                        df, target, feature_columns
                    )
                    
                    # Обучение модели
                    model = train_model(X_train, y_train, task_type)
                    
                    # Получение прогнозов
                    predictions = model.predict(X_test)
                    
                    # Оценка качества
                    metrics = evaluate_model(model, X_test, y_test, task_type)
                    
                    # Вывод метрик
                    st.subheader("Метрики качества модели:")
                    for metric, value in metrics.items():
                        if metric != 'Report':
                            st.metric(metric, value)
                    
                    # Визуализация результатов
                    st.subheader("Важность признаков:")
                    plot_feature_importance(model, 
                                         pd.get_dummies(df[feature_columns], 
                                                      drop_first=True).columns)
                    
                    st.subheader("Сравнение прогнозов с фактическими значениями:")
                    plot_predictions(y_test, predictions, task_type)
        
        # Вкладка предобработки
        with tabs[4]:
            st.subheader("Предобработка данных")
            
            process_type = st.selectbox("Выберите тип обработки", 
                                     ["Изменение типов данных", 
                                      "Обработка пропусков", 
                                      "Удаление дубликатов"],
                                     key="process_type")
            
            if process_type == "Изменение типов данных":
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox("Выберите столбец", df.columns,
                                        key="change_type_column")
                with col2:
                    new_type = st.selectbox("Выберите новый тип", 
                                         ['int64', 'float64', 'str', 'category'],
                                         key="new_type")
                
                if st.button("Применить"):
                    df, success = change_column_type(df, column, new_type)
                    if success:
                        st.session_state['df'] = df
                        st.success("Тип данных успешно изменен")
                        
            elif process_type == "Обработка пропусков":
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox("Выберите столбец", df.columns,
                                        key="missing_column")
                with col2:
                    method = st.selectbox("Выберите метод", 
                                       ['drop', 'fill_value', 'fill_mean', 'fill_median'],
                                       key="missing_method")
                
                value = None
                if method == 'fill_value':
                    value = st.text_input("Введите значение для заполнения")
                
                if st.button("Применить"):
                    df, success = handle_missing_values(df, column, method, value)
                    if success:
                        st.session_state['df'] = df
                        st.success("Пропуски успешно обработаны")
                        
            else:
                if st.button("Удалить дубликаты"):
                    df, success = remove_duplicates(df)
                    if success:
                        st.session_state['df'] = df
                        st.success("Дубликаты успешно удалены")
        
        # Вкладка экспорта
        with tabs[5]:
            st.subheader("Экспорт данных")
            
            format_type = st.selectbox("Выберите формат", ['csv', 'excel'],
                                     key="export_format")
            
            if st.button("Экспортировать"):
                data, mime_type, filename = export_data(df, format_type)
                if data is not None:
                    st.download_button(
                        label="Скачать файл",
                        data=data,
                        file_name=filename,
                        mime=mime_type
                    )

if __name__ == "__main__":
    main()
