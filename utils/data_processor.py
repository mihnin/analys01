import pandas as pd
import streamlit as st
import io

def change_column_type(df, column, new_type):
    """
    Изменение типа данных столбца с проверкой наличия пустых значений
    """
    try:
        if df[column].isnull().any():
            raise ValueError(f"В столбце {column} есть пустые значения. Сначала обработайте их.")
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
        methods = {
            'drop': lambda: df.dropna(subset=[column]),
            'fill_value': lambda: df.assign(**{column: df[column].fillna(value)}),
            'fill_mean': lambda: df.assign(**{column: df[column].fillna(df[column].mean())}),
            'fill_median': lambda: df.assign(**{column: df[column].fillna(df[column].median())})
        }
        return methods[method](), True
    except Exception as e:
        st.error(f"Ошибка при обработке пропущенных значений: {str(e)}")
        return df, False

def remove_duplicates(df, subset=None, keep='first'):
    """
    Удаление дубликатов
    """
    try:
        return df.drop_duplicates(subset=subset, keep=keep), True
    except Exception as e:
        st.error(f"Ошибка при удалении дубликатов: {str(e)}")
        return df, False

def delete_data(df, items, axis=0):
    """
    Универсальная функция удаления строк или столбцов
    axis: 0 для строк, 1 для столбцов
    """
    try:
        return df.drop(items, axis=axis), True
    except Exception as e:
        action = "строк" if axis == 0 else "столбцов"
        st.error(f"Ошибка при удалении {action}: {str(e)}")
        return df, False

def add_computed_column(df, new_column_name, formula):
    """
    Добавление расчетного столбца на основе формулы
    """
    try:
        return df.assign(**{new_column_name: df.eval(formula)}), True
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
            return output.getvalue(), 'text/csv', 'csv'
        elif format_type == 'excel':
            output = io.BytesIO()
            df.to_excel(output, index=False)
            return output.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'xlsx'
        else:
            raise ValueError(f"Неподдерживаемый формат: {format_type}")
    except Exception as e:
        st.error(f"Ошибка при экспорте данных: {str(e)}")
        return None, None, None
