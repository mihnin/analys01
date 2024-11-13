from typing import Optional, Dict, Tuple, List, Any
import pandas as pd
import streamlit as st
from pathlib import Path
import gzip
import zipfile
import logging

def detect_encoding(file_path: str) -> str:
    """Упрощенное определение кодировки файла"""
    encodings = ['utf-8', 'cp1251', 'latin1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read()
                return encoding
        except UnicodeDecodeError:
            continue
    
    return 'utf-8'  # возвращаем utf-8 по умолчанию

def validate_file(file) -> bool:
    """Валидация загружаемого файла"""
    if file is None:
        return False
    
    MAX_FILE_SIZE = 100 * 1024 * 1024
    if file.size > MAX_FILE_SIZE:
        st.error("Файл слишком большой (максимум 100MB)")
        return False
        
    allowed_extensions = {'.csv', '.xlsx', '.xls', '.gz', '.zip'}
    file_ext = Path(file.name).suffix.lower()
    if file_ext not in allowed_extensions:
        st.error("Неподдерживаемый формат файла")
        return False
        
    return True

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Расширенная валидация данных"""
    if df.empty:
        return False, "Пустой датафрейм"
        
    # Проверка на слишком большое количество столбцов
    if len(df.columns) > 1000:
        return False, "Слишком много столбцов (>1000)"
        
    # Проверка на дубликаты имен столбцов
    if len(df.columns) != len(set(df.columns)):
        return False, "Обнаружены дублирующиеся имена столбцов"
        
    return True, ""

def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Автоматическое определение и парсинг столбцов с датой
    """
    # Список возможных форматов даты
    date_formats = [
        '%Y-%m-%d', '%d.%m.%Y', '%m/%d/%Y', 
        '%Y/%m/%d', '%d-%m-%Y', '%m-%д-%Y',
        '%Y-%м-%д %H:%M:%S', '%d.%м.%Y %H:%М:%S'
    ]
    
    # Поиск столбцов с потенциальными датами
    for col in df.columns:
        if df[col].dtype == 'object':
            for date_format in date_formats:
                try:
                    # Попытка распарсить столбец как дату
                    df[col] = pd.to_datetime(df[col], format=date_format)
                    st.info(f"Столбец '{col}' распознан как дата в формате {date_format}")
                    break
                except (ValueError, TypeError):
                    continue
    
    return df

def load_data(file, file_type: str, **kwargs) -> Optional[pd.DataFrame]:
    """Улучшенная загрузка данных"""
    try:
        if not validate_file(file):
            return None
            
        # Определение типа файла по расширению
        if file.name.endswith('.gz'):
            with gzip.open(file) as gz_file:
                df = pd.read_csv(gz_file, **kwargs)
        elif file.name.endswith('.zip'):
            with zipfile.ZipFile(file) as zip_file:
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV files found in ZIP archive")
                with zip_file.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, **kwargs)
        else:
            df = pd.read_csv(file, **kwargs) if file_type == 'csv' else pd.read_excel(file, **kwargs)
            
        # Валидация загруженных данных
        is_valid, error_message = validate_dataframe(df)
        if not is_valid:
            st.error(f"Ошибка валидации данных: {error_message}")
            return None
        
        # Автоматический парсинг столбцов с датами
        df = parse_datetime_columns(df)
            
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        st.error(f"Ошибка при загрузке файла: {str(e)}")
        return None

def get_file_uploader():
    """
    Создание интерфейса загрузки файла
    """
    st.subheader("Загрузка данных")
    
    file = st.file_uploader("Выберите файл CSV или Excel", type=['csv', 'xlsx', 'xls', 'gz', 'zip'])
    
    if file is not None:
        file_type = 'csv' if file.name.endswith('.csv') else 'excel'
        
        with st.expander("Параметры загрузки"):
            if file_type == 'csv':
                separator = st.text_input("Разделитель", value=",")
                encoding = st.selectbox("Кодировка", ['utf-8', 'cp1251', 'latin1'])
                
                # Добавляем выбор столбца даты
                st.subheader("Настройка даты")
                date_column = st.text_input("Столбец даты (необязательно)", value="")
                date_format = st.text_input("Формат даты (необязательно)", value="")
                
                # Параметры для парсинга даты
                parse_dates = [date_column] if date_column else None
                date_parser = None
                if date_format:
                    date_parser = lambda x: pd.to_datetime(x, format=date_format)
                
                df = load_data(
                    file, 
                    file_type, 
                    sep=separator, 
                    encoding=encoding,
                    parse_dates=parse_dates,
                    date_parser=date_parser
                )
            else:
                sheet_name = st.text_input("Имя листа (оставьте пустым для первого листа)", value="")
                kwargs = {'sheet_name': sheet_name} if sheet_name else {}
                df = load_data(file, file_type, **kwargs)
                
            return df
    return None

def load_test_data():
    """Загрузка тестового набора данных"""
    test_file = Path('test_data.csv')
    if not test_file.exists():
        st.error("Тестовый файл не найден")
        logging.error("Test file not found: test_data.csv")
        return None
    try:
        df = pd.read_csv(test_file)
        return df
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        st.error(f"Ошибка при загрузке тестовых данных: {str(e)}")
        return None
