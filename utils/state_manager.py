import streamlit as st
from datetime import datetime
from typing import Optional, Dict

def initialize_session_state():
    """Инициализация состояния сессии"""
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Обзор"
    if 'df' not in st.session_state:
        st.session_state['df'] = None

def save_current_state():
    """Сохранение текущего состояния анализа"""
    try:
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
        st.session_state.last_save_time = datetime.now()
    except Exception as e:
        st.error(f"Ошибка при сохранении состояния: {str(e)}")
