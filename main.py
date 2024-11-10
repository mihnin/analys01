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
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "Обзор"

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
    try:
        # Проверяем, прошло ли достаточно времени с последнего сохранения
        current_time = datetime.now()
        if 'last_save_time' in st.session_state:
            time_diff = (current_time - st.session_state.last_save_time).total_seconds()
            if time_diff < 1:  # Минимальный интервал между сохранениями
                return

        state = {
            'viz_settings': {
                'viz_type': st.session_state.get('viz_type', 'Гистограмма'),
                'viz_column': st.session_state.get('viz_column'),
                'scatter_x': st.session_state.get('scatter_x'),
                'scatter_y': st.session_state.get('scatter_y'),
                'corr_matrix_shown': st.session_state.get('corr_matrix_shown', False)
            },
            'process_settings': {
                'process_type': st.session_state.get('process_type', 'Изменение типов данных'),
                'change_type_column': st.session_state.get('change_type_column'),
                'new_type': st.session_state.get('new_type', 'int64'),
                'missing_column': st.session_state.get('missing_column'),
                'missing_method': st.session_state.get('missing_method', 'drop'),
                'fill_value': st.session_state.get('fill_value')
            },
            'analysis_settings': {
                'outliers_column': st.session_state.get('outliers_column'),
                'selected_stats_column': st.session_state.get('selected_stats_column')
            },
            'report_settings': {
                'selected_sections': st.session_state.get('selected_sections', 
                    ["Базовая информация", "Типы данных", "Статистика"])
            },
            'active_tab': st.session_state.active_tab
        }
        if save_analysis_state('main_app', state):
            st.session_state.last_save_time = current_time
    except Exception as e:
        st.error(f"Ошибка при сохранении состояния: {str(e)}")

def load_saved_state():
    """
    Загрузка сохраненного состояния анализа
    """
    try:
        state = load_analysis_state('main_app')
        if state:
            # Visualization settings
            viz_settings = state.get('viz_settings', {})
            st.session_state['viz_type'] = viz_settings.get('viz_type', 'Гистограмма')
            st.session_state['viz_column'] = viz_settings.get('viz_column')
            st.session_state['scatter_x'] = viz_settings.get('scatter_x')
            st.session_state['scatter_y'] = viz_settings.get('scatter_y')
            st.session_state['corr_matrix_shown'] = viz_settings.get('corr_matrix_shown', False)
            
            # Processing settings
            process_settings = state.get('process_settings', {})
            st.session_state['process_type'] = process_settings.get('process_type', 'Изменение типов данных')
            st.session_state['change_type_column'] = process_settings.get('change_type_column')
            st.session_state['new_type'] = process_settings.get('new_type', 'int64')
            st.session_state['missing_column'] = process_settings.get('missing_column')
            st.session_state['missing_method'] = process_settings.get('missing_method', 'drop')
            st.session_state['fill_value'] = process_settings.get('fill_value')
            
            # Analysis settings
            analysis_settings = state.get('analysis_settings', {})
            st.session_state['outliers_column'] = analysis_settings.get('outliers_column')
            st.session_state['selected_stats_column'] = analysis_settings.get('selected_stats_column')
            
            # Report settings
            report_settings = state.get('report_settings', {})
            st.session_state['selected_sections'] = report_settings.get('selected_sections', 
                ["Базовая информация", "Типы данных", "Статистика"])
            
            # Active tab
            st.session_state.active_tab = state.get('active_tab', 0)
            
            return True
    except Exception as e:
        st.error(f"Ошибка при загрузке состояния: {str(e)}")
    return False

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

    # Загрузка сохраненного состояния при первом запуске
    if not st.session_state.state_loaded:
        if load_saved_state():
            st.session_state.state_loaded = True

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
        
        # Получаем информацию о таблице
        table_info = get_table_info()
        source_name = get_source_name(table_info['source'] if table_info else 'unknown')
        
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
            st.session_state.selected_tab = st.radio(
                "Навигация",
                tab_names,
                index=st.session_state.active_tab,
                key='nav_radio'
            )
            
            # Обновление активной вкладки только при изменении
            if st.session_state.selected_tab is not None:
                new_tab_index = tab_names.index(st.session_state.selected_tab)
                if new_tab_index != st.session_state.active_tab:
                    st.session_state.active_tab = new_tab_index
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
                        st.success("✅ Пропуски успешно обработаны")
                        save_current_state()
                        
            else:  # Удаление дубликатов
                if st.button("Удалить дубликаты"):
                    df, success = remove_duplicates(df)
                    if success:
                        st.session_state['df'] = df
                        save_dataframe(df)
                        st.success("✅ Дубликаты успешно удалены")
                        save_current_state()
        
        # Вкладка экспорта
        with tabs[4]:
            st.subheader("Экспорт данных")
            
            format_type = st.selectbox(
                "Выберите формат экспорта",
                ['csv', 'excel']
            )
            
            if st.button("Экспортировать"):
                try:
                    result = export_data(df, format_type)
                    if result and len(result) == 3:
                        file_content, mime_type, _ = result
                        file_ext = 'csv' if format_type == 'csv' else 'xlsx'
                        if file_content:
                            st.download_button(
                                label="⬇️ Скачать файл",
                                data=file_content,
                                file_name=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                                mime=mime_type
                            )
                except Exception as e:
                    st.error(f"Ошибка при экспорте данных: {str(e)}")

        # Вкладка базы данных
        with tabs[5]:
            st.subheader("Управление базой данных")
            
            if table_info:
                st.write(f"**Текущий источник данных:** {source_name}")
                st.write(f"**Количество строк:** {table_info['rows']}")
                st.write(f"**Размер базы данных:** {table_info['size']} МБ")
                st.write(f"**Последнее обновление:** {format_datetime(datetime.fromisoformat(table_info['last_update']))}")
            
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
                default=st.session_state.get('selected_sections', 
                    ["Базовая информация", "Типы данных", "Статистика"]),
                key="selected_sections"
            )
            
            if st.button("📄 Сгенерировать отчет"):
                if report_sections:
                    try:
                        pdf_data = generate_data_report(df, sections=report_sections)
                        if pdf_data:
                            st.download_button(
                                label="⬇️ Скачать отчет",
                                data=pdf_data,
                                file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
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