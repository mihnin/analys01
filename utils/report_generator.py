import streamlit as st
import pandas as pd
import numpy as np
from fpdf.fpdf import FPDF
import io
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = self._prepare_datatypes(df)
        self.pdf = FPDF()
        self.pdf.add_page()
        
        # Add Unicode font support
        self.pdf.add_font('DejaVu', '', '')
        self.pdf.set_font('DejaVu', '', 12)
        
        # Enable auto page break
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def _prepare_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка типов данных для совместимости с Arrow"""
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

    def add_title(self, title):
        """Добавление заголовка в отчет"""
        try:
            self.pdf.set_font('DejaVu', '', 16)
            self.pdf.cell(0, 10, txt=title, ln=True, align='C')
            self.pdf.set_font('DejaVu', '', 12)
            self.pdf.ln(5)
        except Exception as e:
            st.error(f"Ошибка при добавлении заголовка: {str(e)}")

    def add_section(self, title):
        """Добавление подзаголовка раздела"""
        try:
            self.pdf.set_font('DejaVu', '', 14)
            self.pdf.cell(0, 10, txt=title, ln=True)
            self.pdf.set_font('DejaVu', '', 12)
            self.pdf.ln(2)
        except Exception as e:
            st.error(f"Ошибка при добавлении раздела: {str(e)}")

    def add_text(self, text):
        """Добавление текста в отчет"""
        try:
            # Convert to string and handle encoding
            text = str(text).encode('utf-8', errors='ignore').decode('utf-8')
            self.pdf.multi_cell(0, 10, txt=text)
            self.pdf.ln(2)
        except Exception as e:
            st.error(f"Ошибка при добавлении текста: {str(e)}")

    def add_basic_info(self):
        """Добавление базовой информации о датасете"""
        try:
            self.add_section("Базовая информация")
            info_text = (
                f"Количество строк: {self.df.shape[0]}\n"
                f"Количество столбцов: {self.df.shape[1]}\n"
                f"Размер данных (МБ): {round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)}"
            )
            self.add_text(info_text)
        except Exception as e:
            st.error(f"Ошибка при добавлении базовой информации: {str(e)}")

    def add_data_types(self):
        """Добавление информации о типах данных"""
        try:
            self.add_section("Типы данных")
            dtypes_info = []
            for col, dtype in self.df.dtypes.items():
                nulls = self.df[col].isnull().sum()
                null_percent = round(nulls / len(self.df) * 100, 2)
                dtypes_info.append(f"{col}: {dtype} (пропуски: {nulls}, {null_percent}%)")
            self.add_text("\n".join(dtypes_info))
        except Exception as e:
            st.error(f"Ошибка при добавлении информации о типах данных: {str(e)}")

    def add_numerical_stats(self):
        """Добавление статистики числовых данных"""
        try:
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                self.add_section("Статистика числовых данных")
                stats_df = self.df[numerical_cols].describe()
                stats_text = []
                for col in numerical_cols:
                    stats_text.append(f"\nСтатистика для {col}:")
                    for stat, value in stats_df[col].items():
                        stats_text.append(f"{stat}: {round(value, 2)}")
                self.add_text("\n".join(stats_text))
        except Exception as e:
            st.error(f"Ошибка при добавлении числовой статистики: {str(e)}")

    def add_missing_values(self):
        """Добавление информации о пропущенных значениях"""
        try:
            missing_data = self.df.isnull().sum()
            if missing_data.sum() > 0:
                self.add_section("Пропущенные значения")
                missing_text = []
                for col, count in missing_data[missing_data > 0].items():
                    percent = round(count / len(self.df) * 100, 2)
                    missing_text.append(f"{col}: {count} ({percent}%)")
                self.add_text("\n".join(missing_text))
        except Exception as e:
            st.error(f"Ошибка при добавлении информации о пропущенных значениях: {str(e)}")

    def add_duplicates_info(self):
        """Добавление информации о дубликатах"""
        try:
            duplicates = self.df.duplicated().sum()
            self.add_section("Дубликаты")
            dupl_text = (
                f"Количество дубликатов: {duplicates}\n"
                f"Процент дубликатов: {round(duplicates / len(self.df) * 100, 2)}%"
            )
            self.add_text(dupl_text)
        except Exception as e:
            st.error(f"Ошибка при добавлении информации о дубликатах: {str(e)}")

    def generate_report(self, sections=None, fname: str = None):
        """Генерация полного отчета"""
        try:
            logger.info(f"Начало генерации отчета. Параметры: fname={fname}, sections={sections}")
            
            if fname is None:
                raise ValueError("Параметр 'fname' обязателен для создания отчета")
            
            logger.debug(f"Проверка DataFrame: shape={self.df.shape}, dtypes={self.df.dtypes}")
            
            # Проверяем, что данные готовы к обработке
            if self.df is None:
                raise ValueError("Датафрейм не был корректно инициализирован")
                
            # Добавление заголовка и даты
            self.add_title("Отчет по анализу данных")
            self.add_text(f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
            
            # Если секции не указаны, генерируем полный отчет
            if sections is None:
                sections = ["Базовая информация", "Типы данных", "Статистика", 
                           "Пропущенные значения", "Дубликаты"]
            
            # Добавление выбранных разделов с информированием
            for section in sections:
                logger.info(f"Обработка секции: {section}")
                if section == "Базовая информация":
                    logger.debug(f"Добавление базовой информации: rows={self.df.shape[0]}, cols={self.df.shape[1]}")
                    self.add_basic_info()
                elif section == "Типы данных":
                    self.add_data_types()
                elif section == "Статистика":
                    self.add_numerical_stats()
                elif section == "Пропущенные значения":
                    self.add_missing_values()
                elif section == "Дубликаты":
                    self.add_duplicates_info()

            logger.info(f"Сохранение отчета в файл: {fname}")
            # Сохранение PDF в файл
            try:
                self.pdf.output(fname)
                logging.info(f"Отчет успешно сохранен: {fname}")
                return fname
            except Exception as e:
                error_msg = f"Ошибка при создании PDF: {str(e)}"
                logging.error(error_msg)
                st.error(error_msg)
                return None
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
            fname = f'data_analysis_report_{timestamp}.pdf'
        
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
