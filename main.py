import streamlit as st
from utils.logging_config import setup_logging
from utils.data_loader import load_test_data, get_file_uploader
from utils.state_manager import initialize_session_state, save_current_state
from utils.tab_handlers import (
    show_overview_tab, show_analysis_tab, show_visualization_tab,
    show_preprocessing_tab, show_export_tab, show_reports_tab
)
from utils.report_generator import generate_data_report

# Константы
TAB_NAMES = ["Обзор", "Анализ", "Визуализация", "Предобработка",
             "Экспорт", "Отчеты"]
TAB_HANDLERS = {
    "Обзор": show_overview_tab,
    "Анализ": show_analysis_tab,
    "Визуализация": show_visualization_tab,
    "Предобработка": show_preprocessing_tab,
    "Экспорт": show_export_tab,
    "Отчеты": show_reports_tab
}

logger = setup_logging()

def is_valid_dataframe(df):
    """Проверка валидности датафрейма"""
    return df is not None and not df.empty

def load_data_section():
    """Секция загрузки данных"""
    st.subheader("Загрузка данных")
    
    if st.button("📥 Загрузить тестовые данные"):
        test_df = load_test_data()
        if is_valid_dataframe(test_df):
            st.session_state['df'] = test_df
            save_current_state()
                
    uploaded_df = get_file_uploader()
    if is_valid_dataframe(uploaded_df):
        st.session_state['df'] = uploaded_df
        save_current_state()
        st.success("✅ Данные успешно загружены")

def show_navigation_and_content(df):
    """Отображение навигации и контента"""
    with st.sidebar:
        st.session_state.active_tab = st.radio(
            "Навигация",
            TAB_NAMES,
            index=TAB_NAMES.index(st.session_state.active_tab)
        )

    handler = TAB_HANDLERS.get(st.session_state.active_tab)
    if handler:
        try:
            handler(df)
        except Exception as e:
            logger.error(f"Ошибка в обработчике {st.session_state.active_tab}: {str(e)}")
            st.error(f"Произошла ошибка при отображении {st.session_state.active_tab}")

def main():
    st.set_page_config(
        page_title="Анализ данных",
        page_icon="📊",
        layout="wide"
    )

    try:
        initialize_session_state()
        
        st.title("📊 Комплексный анализ данных")
        
        load_data_section()
        
        if is_valid_dataframe(st.session_state['df']):
            show_navigation_and_content(st.session_state['df'])
            
    except Exception as e:
        logger.error(f"Критическая ошибка в приложении: {str(e)}")
        st.error(f"Произошла критическая ошибка в приложении: {str(e)}")

if __name__ == "__main__":
    main()
