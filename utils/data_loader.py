import pandas as pd
import streamlit as st

def load_data(file, file_type, **kwargs):
    """
    Загрузка данных из файла CSV или Excel
    """
    try:
        if file_type == 'csv':
            return pd.read_csv(file, **kwargs)
        elif file_type == 'excel':
            return pd.read_excel(file, **kwargs)
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        return None

def get_file_uploader():
    """
    Создание интерфейса загрузки файла
    """
    st.subheader("Загрузка данных")
    
    file = st.file_uploader("Выберите файл CSV или Excel", type=['csv', 'xlsx', 'xls'])
    
    if file is not None:
        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
        
        with st.expander("Параметры загрузки"):
            if file_type == 'csv':
                separator = st.text_input("Разделитель", value=",")
                encoding = st.selectbox("Кодировка", ['utf-8', 'cp1251', 'latin1'])
                df = load_data(file, file_type, sep=separator, encoding=encoding)
            else:
                sheet_name = st.text_input("Имя листа (оставьте пустым для первого листа)", value="")
                kwargs = {'sheet_name': sheet_name} if sheet_name else {}
                df = load_data(file, file_type, **kwargs)
                
            return df
    return None
