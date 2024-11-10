import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
import os

def init_db():
    """
    Инициализация базы данных
    """
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        # Создаем таблицу для хранения метаданных
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                table_name TEXT PRIMARY KEY,
                last_update TIMESTAMP,
                source TEXT
            )
        ''')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Ошибка при инициализации базы данных: {str(e)}")
        return False

def save_dataframe(df, table_name='current_data', source='unknown'):
    """
    Сохранение DataFrame в базу данных
    """
    try:
        conn = sqlite3.connect('data.db')
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Обновляем метаданные
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (table_name, last_update, source)
            VALUES (?, ?, ?)
        ''', (table_name, datetime.now().isoformat(), source))
        
        conn.commit()
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
        cursor.execute('DELETE FROM metadata WHERE table_name = ?', (table_name,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Ошибка при удалении данных: {str(e)}")
        return False

def get_table_info(table_name='current_data'):
    """
    Получение информации о таблице
    """
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        
        # Проверяем существование таблицы
        cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone()[0] == 0:
            return None
            
        # Получаем количество строк
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Получаем размер файла БД
        db_size = os.path.getsize('data.db')
        
        # Получаем метаданные
        cursor.execute("SELECT last_update, source FROM metadata WHERE table_name = ?", (table_name,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            last_update, source = result
            return {
                'rows': row_count,
                'size': round(db_size / 1024 / 1024, 2),  # в МБ
                'last_update': last_update,
                'source': source
            }
        return None
    except Exception as e:
        st.error(f"Ошибка при получении информации о таблице: {str(e)}")
        return None

def get_last_update(table_name='current_data'):
    """
    Получение времени последнего обновления
    """
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT last_update FROM metadata WHERE table_name = ?", (table_name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return datetime.fromisoformat(result[0])
        return None
    except Exception as e:
        st.error(f"Ошибка при получении времени последнего обновления: {str(e)}")
        return None
