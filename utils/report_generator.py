import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import io

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация генератора отчетов с DataFrame
        """
        self.df = self._prepare_datatypes(df)

    def _prepare_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка типов данных для совместимости"""
        try:
            # Создаем копию датафрейма
            df_processed = df.copy()
            
            # Конвертируем object колонки в string
            object_columns = df_processed.select_dtypes(include=['object']).columns
            for col in object_columns:
                df_processed[col] = df_processed[col].astype(str)
                
            # Конвертируем category в string
            category_columns = df_processed.select_dtypes(include=['category']).columns
            for col in category_columns:
                df_processed[col] = df_processed[col].astype(str)
                
            logging.info(f"Типы данных успешно преобразованы для {len(object_columns) + len(category_columns)} колонок")
            return df_processed
            
        except Exception as e:
            error_msg = f"Ошибка при подготовке типов данных: {str(e)}"
            logging.error(error_msg)
            st.error(error_msg)
            return df

    def generate_report(self, sections=None, fname: str = None):
        """Генерация полного отчета в Excel"""
        try:
            logger.info(f"Начало генерации отчета. Параметры: fname={fname}, sections={sections}")
            
            if fname is None:
                raise ValueError("Параметр 'fname' обязателен для создания отчета")
            
            # Если секции не указаны, генерируем полный отчет
            if sections is None:
                sections = ["Базовая информация", "Типы данных", "Статистика", 
                           "Пропущенные значения", "Дубликаты"]
            
            # Создание словаря листов для Excel
            excel_sheets = {}
            
            # Базовая информация
            if "Базовая информация" in sections:
                basic_info = pd.DataFrame({
                    'Параметр': ['Количество строк', 'Количество столбцов', 'Размер данных (МБ)'],
                    'Значение': [
                        self.df.shape[0], 
                        self.df.shape[1], 
                        round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                    ]
                })
                excel_sheets['Базовая информация'] = basic_info
            
            # Типы данных
            if "Типы данных" in sections:
                dtypes_info = pd.DataFrame([
                    {'Столбец': col, 
                     'Тип данных': str(dtype), 
                     'Пропуски': self.df[col].isnull().sum(), 
                     'Процент пропусков': round(self.df[col].isnull().sum() / len(self.df) * 100, 2)} 
                    for col, dtype in self.df.dtypes.items()
                ])
                excel_sheets['Типы данных'] = dtypes_info
            
            # Статистика числовых данных
            if "Статистика" in sections:
                numerical_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    stats_df = self.df[numerical_cols].describe()
                    excel_sheets['Статистика'] = stats_df
            
            # Пропущенные значения
            if "Пропущенные значения" in sections:
                missing_data = self.df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Столбец': missing_data.index, 
                    'Количество пропусков': missing_data.values, 
                    'Процент пропусков': round(missing_data / len(self.df) * 100, 2)
                })
                excel_sheets['Пропущенные значения'] = missing_df
            
            # Дубликаты
            if "Дубликаты" in sections:
                duplicates_df = pd.DataFrame({
                    'Параметр': ['Количество дубликатов', 'Процент дубликатов'],
                    'Значение': [
                        self.df.duplicated().sum(), 
                        round(self.df.duplicated().sum() / len(self.df) * 100, 2)
                    ]
                })
                excel_sheets['Дубликаты'] = duplicates_df
            
            # Сохранение в Excel
            logger.info(f"Сохранение отчета в файл: {fname}")
            with pd.ExcelWriter(fname, engine='openpyxl') as writer:
                for sheet_name, df_sheet in excel_sheets.items():
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logging.info(f"Отчет успешно сохранен: {fname}")
            return fname
        
        except Exception as e:
            logger.exception(f"Ошибка при генерации отчета: {str(e)}")
            error_msg = f"Ошибка при генерации отчета: {str(e)}"
            logging.error(error_msg)
            st.error(error_msg)
            return None

def generate_data_report(df: pd.DataFrame, sections=None, fname: str = None) -> str:
    """Создание отчета для датафрейма"""
    logger = logging.getLogger(__name__)
    
    try:
        # Обязательно проверяем и создаём имя файла
        if not fname:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f'data_analysis_report_{timestamp}.xlsx'
        
        # Абсолютный путь для отчета
        reports_dir = Path.cwd() / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Полный путь к файлу отчета
        full_path = str(reports_dir / fname)
        logger.info(f"Создание отчета: {full_path}")
        
        # Создаем генератор отчетов
        generator = ReportGenerator(df)
        
        # Передаем полный путь в generate_report
        result = generator.generate_report(sections=sections, fname=full_path)
        
        if not result:
            logger.error("Не удалось создать отчет")
            return None
            
        if not Path(result).exists():
            logger.error(f"Файл отчета не найден: {result}")
            return None
            
        logger.info(f"Отчет успешно создан: {result}")
        return result
        
    except Exception as e:
        logger.exception(f"Ошибка при создании отчета: {str(e)}")
        return None
