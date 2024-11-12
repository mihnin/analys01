import pandas as pd
import streamlit as st
import io

def change_column_type(df, column, new_type):
    """
    Изменение типа данных столбца
    """
    try:
        df[column] = df[column].astype(new_type)
        return df, True
    except Exception as e:
        st.error(f"Ошибка при изменении типа данных: {str(e)}")
        return df, False

def handle_missing_values(df, column, method, value=None):
    """
    Обработка пропущенных значений
    """
    try:
        if method == 'drop':
            df = df.dropna(subset=[column])
        elif method == 'fill_value':
            df[column] = df[column].fillna(value)
        elif method == 'fill_mean':
            df[column] = df[column].fillna(df[column].mean())
        elif method == 'fill_median':
            df[column] = df[column].fillna(df[column].median())
        return df, True
    except Exception as e:
        st.error(f"Ошибка при обработке пропущенных значений: {str(e)}")
        return df, False

def remove_duplicates(df, subset=None, keep='first'):
    """
    Удаление дубликатов
    """
    try:
        df = df.drop_duplicates(subset=subset, keep=keep)
        return df, True
    except Exception as e:
        st.error(f"Ошибка при удалении дубликатов: {str(e)}")
        return df, False

def delete_rows(df, row_indices):
    """
    Удаление строк по заданным индексам
    """
    try:
        df = df.drop(index=row_indices)
        return df, True
    except Exception as e:
        st.error(f"Ошибка при удалении строк: {str(e)}")
        return df, False

def delete_columns(df, columns):
    """
    Удаление указанных столбцов
    """
    try:
        df = df.drop(columns=columns)
        return df, True
    except Exception as e:
        st.error(f"Ошибка при удалении столбцов: {str(e)}")
        return df, False

def add_computed_column(df, new_column_name, formula):
    """
    Добавление расчетного столбца на основе формулы
    """
    try:
        df[new_column_name] = df.eval(formula)
        return df, True
    except Exception as e:
        st.error(f"Ошибка при добавлении расчетного столбца: {str(e)}")
        return df, False

def export_data(df, format_type):
    """
    Экспорт данных в выбранном формате
    """
    try:
        if format_type == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue(), 'text/csv', 'data.csv'
        elif format_type == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False)
            return output.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'data.xlsx'
    except Exception as e:
        st.error(f"Ошибка при экспорте данных: {str(e)}")
        return None, None, None
