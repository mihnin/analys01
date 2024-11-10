import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import logging.handlers
from typing import Optional, Dict, Any, List, Tuple  # Импортируем все необходимые типы

# Группировка импортов
from utils.data_loader import get_file_uploader
from utils.data_analyzer import (
    get_basic_info, analyze_data_types, analyze_duplicates, 
    get_numerical_stats, analyze_outliers
)
from utils.data_visualizer import (create_histogram, create_box_plot, 
                               create_scatter_plot, plot_correlation_matrix,
                               plot_missing_values, plot_outliers)
from utils.data_processor import (change_column_type, handle_missing_values,
                               remove_duplicates, export_data)
from utils.database import (init_db, save_dataframe, load_dataframe, delete_dataframe,
                          get_table_info, get_last_update, save_analysis_state,
                          load_analysis_state)
from utils.report_generator import generate_data_report

# Настройка расширенного логирования
log_handler = logging.handlers.RotatingFileHandler(
    'app.log',
    maxBytes=1024*1024,
    backupCount=5
)
logging.basicConfig(
    handlers=[log_handler],
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def handle_error(func):
    """Декоратор для обработки ошибок"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"Произошла ошибка: {str(e)}")
            return None
    return wrapper

@handle_error
def load_state() -> Optional[dict]:
    """Загрузка состояния с обработкой ошибок"""
    state = load_analysis_state('main_app')
    if state:
        st.session_state.update(state)
        return state
    return None

def initialize_session_state():
    """
    Инициализация состояния сессии
    """
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'state_loaded' not in st.session_state:
        st.session_state.state_loaded = False
    if 'last_save_time' not in st.session_state:
        st.session_state.last_save_time = datetime.now()

def load_test_data():
    """
    Загрузка тестового набора данных
    """
    test_file = Path('test_data.csv')
    if not test_file.exists():
        st.error("Тестовый файл не найден")
        logging.error("Test file not found: test_data.csv")
        return None
    try:
        df = pd.read_csv(test_file)
        if save_dataframe(df, source='test_data'):
            st.success("✅ Тестовые данные успешно сохранены в базу данных")
            logging.info("Test data successfully loaded and saved")
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        st.error(f"Ошибка при загрузке тестовых данных: {str(e)}")
        return None

def save_current_state():
    """
    Сохранение текущего состояния анализа
    """
    try:
        current_time = datetime.now()
        state = {
            'active_tab': st.session_state.active_tab,
            'viz_settings': {
                'viz_type': st.session_state.get('viz_type', 'Гистограмма'),
                'viz_column': st.session_state.get('viz_column'),
            },
            'process_settings': {
                'process_type': st.session_state.get('process_type', 'Изменение типов данных'),
                'change_type_column': st.session_state.get('change_type_column'),
            }
        }
        if save_analysis_state('main_app', state):
            st.session_state.last_save_time = current_time
    except Exception as e:
        st.error(f"Ошибка при сохранении состояния: {str(e)}")

def main():
    st.set_page_config(
        page_title="Анализ данных",
        page_icon="📊",
        layout="wide"
    )

    # Инициализация состояния сессии
    initialize_session_state()

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
            save_current_state()
            
    # Стандартный загрузчик файлов
    uploaded_df = get_file_uploader()
    if uploaded_df is not None:
        st.session_state['df'] = uploaded_df
        if save_dataframe(uploaded_df, source='file_upload'):
            st.success("✅ Загруженные данные успешно сохранены в базу данных")
            save_current_state()
    
    # Работа с загруженными данными
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Определение вкладок
        tab_names = [
            "Обзор", 
            "Анализ", 
            "Визуализация",
            "Предобработка",
            "Экспорт",
            "База данных",
            "Отчеты"
        ]
        
        # Создание навигации в сайдбаре
        with st.sidebar:
            st.write(f"🔍 Текущая активная вкладка: {st.session_state.active_tab}")
            
            selected_tab = st.radio(
                "Навигация",
                tab_names,
                index=st.session_state.active_tab
            )
            
            # Обновляем активную вкладку при изменении
            current_tab_index = tab_names.index(selected_tab)
            if current_tab_index != st.session_state.active_tab:
                st.session_state.active_tab = current_tab_index
                save_current_state()
        
        # Создание вкладок
        tabs = st.tabs(tab_names)
        
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
                if selected_column:
                    lower_bound, upper_bound = analyze_outliers(df, selected_column)
                    if lower_bound is not None and upper_bound is not None:
                        plot_outliers(df, selected_column, lower_bound, upper_bound)
            else:
                st.info("В датасете нет числовых столбцов для анализа выбросов")
        
        # Вкладка визуализации
        with tabs[2]:
            st.subheader("Визуализация данных")
            
            viz_type = st.selectbox(
                "Выберите тип визуализации", 
                ["Гистограмма", "Box Plot", "Scatter Plot", "Корреляционная матрица"],
                key="viz_type"
            )
            
            if viz_type in ["Гистограмма", "Box Plot"]:
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    column = st.selectbox(
                        "Выберите столбец", 
                        numerical_cols,
                        key="viz_column"
                    )
                    if column:
                        if viz_type == "Гистограмма":
                            create_histogram(df, column)
                        else:
                            create_box_plot(df, column)
                else:
                    st.info("В датасете нет числовых столбцов для визуализации")
                    
            elif viz_type == "Scatter Plot":
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_column = st.selectbox(
                            "Выберите X", 
                            numerical_cols,
                            key="scatter_x"
                        )
                    with col2:
                        y_column = st.selectbox(
                            "Выберите Y", 
                            numerical_cols,
                            key="scatter_y"
                        )
                    if x_column and y_column:
                        create_scatter_plot(df, x_column, y_column)
                else:
                    st.info("В датасете нет числовых столбцов для визуализации")
                
            else:  # Корреляционная матрица
                plot_correlation_matrix(df)
        
        # Вкладка предобработки
        with tabs[3]:
            st.subheader("Предобработка данных")
            
            process_type = st.selectbox(
                "Выберите тип обработки", 
                ["Изменение типов данных", "Обработка пропусков", "Удаление дубликатов"],
                key="process_type"
            )
            
            if process_type == "Изменение типов данных":
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox(
                        "Выберите столбец",
                        df.columns,
                        key="change_type_column"
                    )
                with col2:
                    new_type = st.selectbox(
                        "Выберите новый тип", 
                        ['int64', 'float64', 'str', 'category'],
                        key="new_type"
                    )
                
                if st.button("Применить"):
                    df, success = change_column_type(df, column, new_type)
                    if success:
                        st.session_state['df'] = df
                        save_dataframe(df)
                        st.success("✅ Тип данных успешно изменен")
                        save_current_state()
                        
            elif process_type == "Обработка пропусков":
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox(
                        "Выберите столбец",
                        df.columns,
                        key="missing_column"
                    )
                with col2:
                    method = st.selectbox(
                        "Выберите метод", 
                        ['drop', 'fill_value', 'fill_mean', 'fill_median'],
                        key="missing_method"
                    )
                
                value = None
                if method == 'fill_value':
                    value = st.text_input("Введите значение для заполнения", key="fill_value")
                
                if st.button("Применить"):
                    df, success = handle_missing_values(df, column, method, value)
                    if success:
                        st.session_state['df'] = df
                        save_dataframe(df)
                        st.success("✅ Пропущенные значения обработаны")
                        save_current_state()
                        
            else:  # Удаление дубликатов
                if st.button("Удалить дубликаты"):
                    df, success = remove_duplicates(df)
                    if success:
                        st.session_state['df'] = df
                        save_dataframe(df)
                        st.success("✅ Дубликаты удалены")
                        save_current_state()
        
        # Вкладка экспорта
        with tabs[4]:
            st.subheader("Экспорт данных")
            
            format_type = st.selectbox(
                "Выберите формат экспорта",
                ['csv', 'excel']
            )
            
            if st.button("Экспортировать"):
                result = export_data(df, format_type)
                if result and len(result) == 3:
                    data, mime_type, _ = result
                    file_ext = format_type
                    if data:
                        st.download_button(
                            label="⬇️ Скачать файл",
                            data=data,
                            file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type
                        )
        
        # Вкладка базы данных
        with tabs[5]:
            st.subheader("Управление базой данных")
            
            table_info = get_table_info()
            if table_info:
                st.write(f"**Текущий источник данных:** {table_info['source']}")
                st.write(f"**Количество строк:** {table_info['rows']}")
                st.write(f"**Размер базы данных:** {table_info['size']} МБ")
                st.write(f"**Последнее обновление:** {table_info['last_update']}")
            
            if st.button("❌ Очистить базу данных"):
                if delete_dataframe():
                    st.session_state.pop('df', None)
                    st.success("✅ База данных успешно очищена")
                    st.rerun()
        
        # Вкладка отчетов
        with tabs[6]:
            st.subheader("Генерация отчетов")
            
            report_sections = st.multiselect(
                "Выберите разделы для отчета",
                ["Базовая информация", "Типы данных", "Статистика", 
                 "Пропущенные значения", "Дубликаты"],
                default=["Базовая информация", "Типы данных", "Статистика"]
            )
            
            if st.button("📄 Сгенерировать отчет"):
                if report_sections:
                    try:
                        report_filename = generate_data_report(df, sections=report_sections, fname='data_analysis_report.pdf')
                        if report_filename:
                            with open(report_filename, 'rb') as file:
                                st.download_button(
                                    label="⬇️ Скачать отчет",
                                    data=file,
                                    file_name=report_filename,
                                    mime="application/pdf"
                                )
                            st.success("✅ Отчет успешно сгенерирован")
                            save_current_state()
                    except Exception as e:
                        st.error(f"Ошибка при генерации отчета: {str(e)}")
                else:
                    st.warning("Выберите хотя бы один раздел для отчета")

if __name__ == "__main__":
    main()
