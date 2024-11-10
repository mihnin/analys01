import sqlite3
import pandas as pd
import streamlit as st

def init_db():
    """
    Инициализация базы данных
    """
    try:
        conn = sqlite3.connect('data.db')
        conn.close()
        return True
    except Exception as e:
        st.error(f"Ошибка при инициализации базы данных: {str(e)}")
        return False

def save_dataframe(df, table_name='current_data'):
    """
    Сохранение DataFrame в базу данных
    """
    try:
        conn = sqlite3.connect('data.db')
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()
        return True
    except Exception as e:
        st.error(f"Ошибка при сохранении данных: {str(e)}")
        return False

def load_dataframe(table_name='current_data'):
    """
    Загрузка DataFrame из базы данных
    """
    try:
        conn = sqlite3.connect('data.db')
        df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

def delete_dataframe(table_name='current_data'):
    """
    Удаление данных из базы данных
    """
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Ошибка при удалении данных: {str(e)}")
        return False
