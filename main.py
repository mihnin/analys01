import streamlit as st
from utils.logging_config import setup_logging
from utils.database import init_db, load_dataframe, save_dataframe
from utils.data_loader import load_test_data, get_file_uploader
from utils.state_manager import initialize_session_state, save_current_state
from utils.tab_handlers import (
    show_overview_tab, show_analysis_tab, show_visualization_tab,
    show_preprocessing_tab, show_export_tab, show_database_tab, show_reports_tab
)

logger = setup_logging()

def main():
    st.set_page_config(
        page_title="Анализ данных",
        page_icon="📊",
        layout="wide"
    )

    # Инициализация
    initialize_session_state()
    if not init_db():
        st.error("Не удалось инициализировать базу данных")
        return

    st.title("📊 Комплексный анализ данных")
    
    # Загрузка данных из БД
    if st.session_state['df'] is None:
        df = load_dataframe()
        if df is not None:
            st.session_state['df'] = df
            st.success("✅ Данные успешно загружены из базы данных")
    
    # Секция загрузки данных
    st.subheader("Загрузка данных")
    
    if st.button("📥 Загрузить тестовые данные"):
        test_df = load_test_data()
        if test_df is not None:
            st.session_state['df'] = test_df
            save_current_state()
            
    uploaded_df = get_file_uploader()
    if uploaded_df is not None:
        st.session_state['df'] = uploaded_df
        if save_dataframe(uploaded_df, source='file_upload'):
            st.success("✅ Загруженные данные успешно сохранены в базу данных")
            save_current_state()
    
    # Работа с загруженными данными
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        # Навигация
        tab_names = ["Обзор", "Анализ", "Визуализация", "Предобработка", 
                    "Экспорт", "База данных", "Отчеты"]
        
        with st.sidebar:
            st.session_state.active_tab = st.radio(
                "Навигация",
                tab_names,
                index=tab_names.index(st.session_state.active_tab)
            )

        # Отображение вкладок
        tab_handlers = {
            "Обзор": show_overview_tab,
            "Анализ": show_analysis_tab,
            "Визуализация": show_visualization_tab,
            "Предобработка": show_preprocessing_tab,
            "Экспорт": show_export_tab,
            "База данных": show_database_tab,
            "Отчеты": show_reports_tab
        }
        
        if handler := tab_handlers.get(st.session_state.active_tab):
            handler(df)

if __name__ == "__main__":
    main()
