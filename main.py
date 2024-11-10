import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
from utils.database import (init_db, save_dataframe, load_dataframe, delete_dataframe,
                          get_table_info, get_last_update)

def load_test_data():
    """
    Загрузка тестового набора данных
    """
    try:
        df = pd.read_csv('test_data.csv')
        if save_dataframe(df, source='test_data'):
            st.success("✅ Тестовые данные успешно сохранены в базу данных")
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке тестовых данных: {str(e)}")
        return None

def format_datetime(dt):
    """
    Форматирование даты и времени
    """
    if dt:
        return dt.strftime("%d.%m.%Y %H:%M:%S")
    return "Нет данных"

def get_source_name(source):
    """
    Преобразование технического названия источника в понятное пользователю
    """
    source_mapping = {
        'test_data': 'Тестовые данные',
        'file_upload': 'Загруженный файл',
        'unknown': 'Данные'
    }
    return source_mapping.get(source, 'Данные')

def check_data_quality(df, target, features):
    """
    Проверка качества данных перед обучением модели
    """
    # Проверка на пропущенные значения
    missing_target = df[target].isnull().sum()
    missing_features = df[features].isnull().sum()
    
    has_issues = False
    
    if missing_target > 0:
        st.warning(f"⚠️ В целевой переменной '{target}' обнаружено {missing_target} "
                  f"пропущенных значений ({round(missing_target/len(df)*100, 2)}%)")
        has_issues = True
    
    features_with_nulls = missing_features[missing_features > 0]
    if not features_with_nulls.empty:
        st.warning("⚠️ Обнаружены пропущенные значения в признаках:")
        for feature, null_count in features_with_nulls.items():
            st.write(f"- {feature}: {null_count} пропусков "
                    f"({round(null_count/len(df)*100, 2)}%)")
        has_issues = True
    
    if has_issues:
        st.info("ℹ️ Рекомендуется обработать пропущенные значения перед обучением модели. "
                "Используйте вкладку 'Предобработка' для работы с пропущенными значениями.")
    
    return not has_issues

def main():
    st.set_page_config(
        page_title="Анализ данных",
        page_icon="📊",
        layout="wide"
    )

    # Инициализация БД при запуске
    if not init_db():
        st.error("Не удалось инициализировать базу данных")
        return

    st.title("📊 Комплексный анализ данных")
    
    # Секция загрузки данных
    st.subheader("Загрузка данных")
    
    # Попытка загрузить данные из БД при старте
    if 'df' not in st.session_state:
        df = load_dataframe()
        if df is not None:
            st.session_state['df'] = df
            st.success("✅ Данные успешно загружены из базы данных")
    
    # Кнопка загрузки тестовых данных
    if st.button("📥 Загрузить тестовые данные"):
        test_df = load_test_data()
        if test_df is not None:
            st.session_state['df'] = test_df
            st.success("✅ Тестовые данные успешно загружены!")
            
    # Стандартный загрузчик файлов
    uploaded_df = get_file_uploader()
    if uploaded_df is not None:
        st.session_state['df'] = uploaded_df
        if save_dataframe(uploaded_df, source='file_upload'):
            st.success("✅ Загруженные данные успешно сохранены в базу данных")
    
    # Работа с загруженными данными
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Получаем информацию о таблице
        table_info = get_table_info()
        source_name = get_source_name(table_info['source'] if table_info else 'unknown')
        
        # Создание вкладок
        tabs = st.tabs([
            "Обзор", 
            "Анализ", 
            "Визуализация",
            "Прогнозирование", 
            "Предобработка",
            "Экспорт",
            "База данных"
        ])
        
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
            
            # Выбор целевой переменной и признаков
            if task_type == "регрессия":
                target_columns = df.select_dtypes(include=[np.number]).columns
                if len(target_columns) == 0:
                    st.error("❌ В датасете нет числовых столбцов для задачи регрессии")
                    st.stop()
            else:
                target_columns = df.select_dtypes(include=['object', 'category']).columns
                if len(target_columns) == 0:
                    st.error("❌ В датасете нет категориальных столбцов для задачи классификации")
                    st.stop()
                
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
            
            if st.button("Обучить модель"):
                if len(feature_columns) == 0:
                    st.error("❌ Необходимо выбрать хотя бы один признак для обучения модели")
                else:
                    # Проверка качества данных
                    if check_data_quality(df, target, feature_columns):
                        with st.spinner("⏳ Обучение модели..."):
                            try:
                                # Подготовка данных
                                X_train, X_test, y_train, y_test, scaler = prepare_data(
                                    df, target, feature_columns, task_type
                                )
                                
                                # Обучение модели
                                model = train_model(X_train, y_train, task_type)
                                
                                # Получение прогнозов
                                predictions = model.predict(X_test)
                                
                                # Оценка качества
                                metrics = evaluate_model(model, X_test, y_test, task_type)
                                
                                # Вывод метрик
                                st.subheader("📊 Метрики качества модели:")
                                for metric, value in metrics.items():
                                    if metric != 'Report':
                                        st.metric(metric, value)
                                
                                # Визуализация результатов
                                st.subheader("📈 Важность признаков:")
                                plot_feature_importance(model, feature_columns)
                                
                                st.subheader("🎯 Сравнение прогнозов с фактическими значениями:")
                                plot_predictions(y_test, predictions, task_type)
                                
                                st.success("✅ Модель успешно обучена!")
                            except Exception as e:
                                st.error(f"❌ Ошибка при обучении модели: {str(e)}")
        
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
                        save_dataframe(df)  # Сохраняем изменения в БД
                        st.success("✅ Тип данных успешно изменен")
                        
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
                        save_dataframe(df)  # Сохраняем изменения в БД
                        st.success("✅ Пропуски успешно обработаны")
                        
            else:
                if st.button("Удалить дубликаты"):
                    df, success = remove_duplicates(df)
                    if success:
                        st.session_state['df'] = df
                        save_dataframe(df)  # Сохраняем изменения в БД
                        st.success("✅ Дубликаты успешно удалены")
        
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

        # Вкладка базы данных
        with tabs[6]:
            st.subheader("Состояние базы данных")
            
            # Информация о подключении
            st.info("🔌 База данных подключена и готова к работе")
            
            # Информация о текущей таблице
            if table_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Количество строк", table_info['rows'])
                with col2:
                    st.metric("Размер БД (МБ)", table_info['size'])
                with col3:
                    st.metric("Источник данных", get_source_name(table_info['source']))
                with col4:
                    st.metric("Последнее обновление", 
                             format_datetime(datetime.fromisoformat(table_info['last_update'])))
            else:
                st.warning("⚠️ В базе данных нет сохраненных таблиц")
            
            # Кнопка очистки БД
            if st.button("🗑️ Очистить базу данных"):
                if delete_dataframe():
                    st.success("✅ База данных успешно очищена")
                    if 'df' in st.session_state:
                        del st.session_state['df']
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
