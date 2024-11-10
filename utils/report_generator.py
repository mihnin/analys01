import streamlit as st
import pandas as pd
import numpy as np
from fpdf.fpdf import FPDF
import io
from datetime import datetime

class ReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pdf = FPDF()
        self.pdf.add_page()
        
        # Add Unicode font support
        self.pdf.add_font('DejaVu', '', '')
        self.pdf.set_font('DejaVu', '', 12)
        
        # Enable auto page break
        self.pdf.set_auto_page_break(auto=True, margin=15)

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

    def generate_report(self, sections=None, fname='report.pdf'):
        """Генерация полного отчета"""
        try:
            # Добавление заголовка и даты
            self.add_title("Отчет по анализу данных")
            self.add_text(f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
            
            # Если секции не указаны, генерируем полный отчет
            if sections is None:
                sections = ["Базовая информация", "Типы данных", "Статистика", 
                           "Пропущенные значения", "Дубликаты"]
            
            # Добавление выбранных разделов
            if "Базовая информация" in sections:
                self.add_basic_info()
            if "Типы данных" in sections:
                self.add_data_types()
            if "Статистика" in sections:
                self.add_numerical_stats()
            if "Пропущенные значения" in sections:
                self.add_missing_values()
            if "Дубликаты" in sections:
                self.add_duplicates_info()

            # Сохранение PDF в файл
            try:
                self.pdf.output(fname)
                return fname
            except Exception as e:
                st.error(f"Ошибка при создании PDF: {str(e)}")
                return None
        except Exception as e:
            st.error(f"Ошибка при генерации отчета: {str(e)}")
            return None

def generate_data_report(df: pd.DataFrame, sections=None, fname='report.pdf') -> str:
    """Создание отчета для датафрейма"""
    try:
        generator = ReportGenerator(df)
        return generator.generate_report(sections=sections, fname=fname)
    except Exception as e:
        st.error(f"Ошибка при создании отчета: {str(e)}")
        return None
