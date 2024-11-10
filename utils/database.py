import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
import json
import os
import contextlib
from typing import Optional, Dict, Any, Tuple, List
import queue

class DatabasePool:
    """Пул соединений с базой данных"""
    def __init__(self, max_connections: int = 5):
        self.pool = queue.Queue(maxsize=max_connections)
        for _ in range(max_connections):
            self.pool.put(sqlite3.connect('data.db', check_same_thread=False))

    def get_connection(self) -> sqlite3.Connection:
        return self.pool.get()

    def return_connection(self, connection: sqlite3.Connection):
        self.pool.put(connection)

db_pool = DatabasePool()

@contextlib.contextmanager
def get_db_connection():
    """Улучшенный контекстный менеджер для работы с БД"""
    conn = None
    try:
        conn = db_pool.get_connection()
        yield conn
    finally:
        if conn:
            db_pool.return_connection(conn)

def execute_query(query: str, params: tuple = (), fetch: bool = False) -> Optional[Any]:
    """Безопасное выполнение SQL-запросов"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            conn.commit()
            return True
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        return None

def init_db():
    """
    Инициализация базы данных
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Создаем таблицу для хранения метаданных
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    table_name TEXT PRIMARY KEY,
                    last_update TIMESTAMP,
                    source TEXT
                )
            ''')
            # Создаем таблицу для хранения состояния анализа
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_state (
                    id INTEGER PRIMARY KEY,
                    component_name TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Ошибка при инициализации базы данных: {str(e)}")
        return False

def save_analysis_state(component_name: str, state_data: dict):
    """
    Сохранение состояния анализа
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            state_json = json.dumps(state_data)
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_state (component_name, state_data, last_update)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (component_name, state_json))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Ошибка при сохранении состояния: {str(e)}")
        return False

def load_analysis_state(component_name: str) -> dict:
    """
    Загрузка состояния анализа
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT state_data FROM analysis_state WHERE component_name = ?', 
                          (component_name,))
            result = cursor.fetchone()
        
        if result:
            return json.loads(result[0])
        return {}
    except Exception as e:
        st.error(f"Ошибка при загрузке состояния: {str(e)}")
        return {}

def save_dataframe(df: pd.DataFrame, table_name: str = 'current_data', 
                  source: str = 'unknown') -> bool:
    """Сохранение DataFrame в базу данных"""
    if df is None or df.empty:
        logging.error("Attempt to save empty DataFrame")
        return False
        
    try:
        with get_db_connection() as conn:
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO metadata (table_name, last_update, source)
                VALUES (?, ?, ?)
            ''', (table_name, datetime.now().isoformat(), source))
            conn.commit()
        return True
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        st.error(f"Ошибка при сохранении данных: {str(e)}")
        return False

def load_dataframe(table_name='current_data'):
    """
    Загрузка DataFrame из базы данных
    """
    try:
        with get_db_connection() as conn:
            df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {str(e)}")
        return None

def delete_dataframe(table_name='current_data'):
    """
    Удаление данных из базы данных
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f'DROP TABLE IF EXISTS {table_name}')
            cursor.execute('DELETE FROM metadata WHERE table_name = ?', (table_name,))
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Ошибка при удалении данных: {str(e)}")
        return False

def get_table_info(table_name='current_data'):
    """
    Получение информации о таблице
    """
    try:
        with get_db_connection() as conn:
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
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_update FROM metadata WHERE table_name = ?", (table_name,))
            result = cursor.fetchone()
        
        if result:
            return datetime.fromisoformat(result[0])
        return None
    except Exception as e:
        st.error(f"Ошибка при получении времени последнего обновления: {str(e)}")
        return None
