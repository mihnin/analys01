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
from utils.database import (init_db, save_dataframe, load_dataframe, delete_dataframe,
                           get_table_info, get_last_update, save_analysis_state,
                           load_analysis_state)
from utils.report_generator import generate_data_report

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

def save_current_state():
    """
    Сохранение текущего состояния анализа
    """
    state = {
        'active_tab': st.session_state.get('active_tab', 0),
        'viz_settings': {
            'viz_type': st.session_state.get('viz_type', 'Гистограмма'),
            'viz_column': st.session_state.get('viz_column', None),
            'scatter_x': st.session_state.get('scatter_x', None),
            'scatter_y': st.session_state.get('scatter_y', None)
        },
        'process_settings': {
            'process_type': st.session_state.get('process_type', 'Изменение типов данных'),
            'change_type_column': st.session_state.get('change_type_column', None),
            'new_type': st.session_state.get('new_type', 'int64'),
            'missing_column': st.session_state.get('missing_column', None),
            'missing_method': st.session_state.get('missing_method', 'drop')
        },
        'outliers_column': st.session_state.get('outliers_column', None)
    }
    save_analysis_state('main_app', state)

def load_saved_state():
    """
    Загрузка сохраненного состояния анализа
    """
    state = load_analysis_state('main_app')
    if state:
        # Восстановление состояния в session_state
        st.session_state['active_tab'] = state.get('active_tab', 0)
        
        viz_settings = state.get('viz_settings', {})
        st.session_state['viz_type'] = viz_settings.get('viz_type', 'Гистограмма')
        st.session_state['viz_column'] = viz_settings.get('viz_column')
        st.session_state['scatter_x'] = viz_settings.get('scatter_x')
        st.session_state['scatter_y'] = viz_settings.get('scatter_y')
        
        process_settings = state.get('process_settings', {})
        st.session_state['process_type'] = process_settings.get('process_type', 'Изменение типов данных')
        st.session_state['change_type_column'] = process_settings.get('change_type_column')
        st.session_state['new_type'] = process_settings.get('new_type', 'int64')
        st.session_state['missing_column'] = process_settings.get('missing_column')
        st.session_state['missing_method'] = process_settings.get('missing_method', 'drop')
        
        st.session_state['outliers_column'] = state.get('outliers_column')

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

    # Загрузка сохраненного состояния
    if 'state_loaded' not in st.session_state:
        load_saved_state()
        st.session_state['state_loaded'] = True

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
        tab_names = [
            "Обзор", 
            "Анализ", 
            "Визуализация",
            "Предобработка",
            "Экспорт",
            "База данных",
            "Отчеты"
        ]
        
        # Используем сохраненный индекс активной вкладки
        active_tab = st.session_state.get('active_tab', 0)
        tabs = st.tabs(tab_names)
        
        # Вкладка обзора
        with tabs[0]:
            if active_tab == 0:
                get_basic_info(df)
                st.dataframe(df.head())
                analyze_data_types(df)
        
        # Вкладка анализа
        with tabs[1]:
            if active_tab == 1:
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
            if active_tab == 2:
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
        
        # Вкладка предобработки
        with tabs[3]:
            if active_tab == 3:
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
                            save_dataframe(df)
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
                            save_dataframe(df)
                            st.success("✅ Пропуски успешно обработаны")
                            
                else:
                    if st.button("Удалить дубликаты"):
                        df, success = remove_duplicates(df)
                        if success:
                            st.session_state['df'] = df
                            save_dataframe(df)
                            st.success("✅ Дубликаты успешно удалены")
        
        # Вкладка экспорта
        with tabs[4]:
            if active_tab == 4:
                st.subheader("Экспорт данных")
                
                format_type = st.selectbox("Выберите формат", ['csv', 'excel'],
                                       key="export_format")
                
                if st.button("Экспортировать"):
                    data, mime_type, file_name = export_data(df, format_type)
                    if data:
                        st.download_button(
                            label="Скачать файл",
                            data=data,
                            file_name=file_name,
                            mime=mime_type
                        )
        
        # Вкладка базы данных
        with tabs[5]:
            if active_tab == 5:
                st.subheader("Информация о базе данных")
                
                if table_info:
                    st.write(f"**Источник данных:** {source_name}")
                    st.write(f"**Последнее обновление:** {format_datetime(datetime.fromisoformat(table_info['last_update']))}")
                    st.write(f"**Количество записей:** {table_info['rows']}")
                    st.write(f"**Размер базы данных:** {table_info['size']} МБ")
                    
                    if st.button("Очистить базу данных"):
                        if delete_dataframe():
                            st.session_state.pop('df', None)
                            st.success("✅ База данных успешно очищена")
                            st.experimental_rerun()
                else:
                    st.info("База данных пуста")
        
        # Вкладка отчетов
        with tabs[6]:
            if active_tab == 6:
                st.subheader("Генерация отчетов")
                
                report_options = st.multiselect(
                    "Выберите разделы для включения в отчет",
                    ["Базовая информация", "Типы данных", "Статистика", 
                     "Пропущенные значения", "Дубликаты"],
                    default=["Базовая информация", "Типы данных", "Статистика"]
                )
                
                if st.button("Сгенерировать отчет"):
                    try:
                        report_bytes = generate_data_report(df, sections=report_options)
                        
                        st.download_button(
                            label="📥 Скачать отчет",
                            data=report_bytes,
                            file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ Отчет успешно сгенерирован!")
                    except Exception as e:
                        st.error(f"Ошибка при генерации отчета: {str(e)}")
        
        # Обновление активной вкладки и сохранение состояния
        for i, tab in enumerate(tabs):
            if tab._is_active():
                st.session_state['active_tab'] = i
                save_current_state()
                break

if __name__ == "__main__":
    main()
