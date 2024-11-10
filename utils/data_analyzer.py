import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
import plotly.figure_factory as ff
from scipy.stats import norm

@st.cache_data
def get_advanced_stats(df, column):
    """
    Расширенный статистический анализ для числового столбца
    """
    data = df[column].dropna()
    
    # Базовые статистики
    stats_dict = {
        'Среднее': data.mean(),
        'Медиана': data.median(),
        'Мода': data.mode().iloc[0] if not data.mode().empty else None,
        'СКО': data.std(),
        'Дисперсия': data.var(),
        'Минимум': data.min(),
        'Максимум': data.max(),
        'Размах': data.max() - data.min(),
    }
    
    # Квартили и межквартильный размах
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    stats_dict.update({
        'Q1 (25%)': q1,
        'Q3 (75%)': q3,
        'IQR': iqr
    })
    
    # Асимметрия и эксцесс
    stats_dict.update({
        'Асимметрия': data.skew(),
        'Эксцесс': data.kurtosis(),
    })
    
    # Доверительный интервал для среднего (95%)
    confidence_level = 0.95
    n = len(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence_level, n-1, loc=data.mean(), scale=std_err)
    stats_dict.update({
        'CI нижняя граница (95%)': ci[0],
        'CI верхняя граница (95%)': ci[1]
    })
    
    return stats_dict

@st.cache_data
def analyze_distribution(df, column):
    """
    Расширенный анализ распределения данных
    """
    st.subheader(f"Анализ распределения: {column}")
    
    data = df[column].dropna()
    
    # Получение расширенных статистик
    stats_dict = get_advanced_stats(df, column)
    
    # Вывод основных статистик в три колонки
    cols = st.columns(3)
    stats_items = list(stats_dict.items())
    for i, col in enumerate(cols):
        with col:
            for key, value in stats_items[i::3]:  # Распределение статистик по колонкам
                if value is not None:
                    st.metric(key, round(value, 4) if isinstance(value, float) else value)
    
    # Тесты на нормальность
    st.subheader("Тесты на нормальность распределения")
    normality_tests = perform_normality_test(data)
    
    cols = st.columns(3)
    with cols[0]:
        if normality_tests['shapiro']:
            st.metric("Тест Шапиро-Уилка (p-value)", 
                     f"{normality_tests['shapiro'][1]:.4f}")
    with cols[1]:
        st.metric("Тест Колмогорова-Смирнова (p-value)", 
                 f"{normality_tests['ks'][1]:.4f}")
    with cols[2]:
        st.metric("Тест D'Agostino K^2 (p-value)", 
                 f"{normality_tests['k2'][1]:.4f}")
    
    # Интерпретация результатов
    st.write("### Интерпретация распределения")
    p_threshold = 0.05
    interpretation = []
    
    # Интерпретация асимметрии
    skew = stats_dict['Асимметрия']
    if abs(skew) < 0.5:
        interpretation.append("✅ Распределение близко к симметричному")
    else:
        interpretation.append("⚠️ Распределение асимметрично " + 
                            ("(правосторонняя асимметрия)" if skew > 0 else "(левосторонняя асимметрия)"))
    
    # Интерпретация эксцесса
    kurtosis = stats_dict['Эксцесс']
    if abs(kurtosis) < 0.5:
        interpretation.append("✅ Эксцесс близок к нормальному распределению")
    else:
        interpretation.append("ℹ️ " + ("Распределение имеет тяжелые хвосты" if kurtosis > 0 
                                     else "Распределение имеет легкие хвосты"))
    
    # Интерпретация тестов на нормальность
    normal_count = sum(1 for test in normality_tests.values() 
                      if test and test[1] > p_threshold)
    
    if normal_count >= 2:
        interpretation.append("✅ Распределение можно считать нормальным")
    else:
        interpretation.append("⚠️ Распределение существенно отличается от нормальн��го")
    
    for interp in interpretation:
        st.write(interp)

def get_basic_info(df):
    """
    Получение базовой информации о датасете
    """
    st.subheader("Обзор данных")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Количество строк", df.shape[0])
    with col2:
        st.metric("Количество столбцов", df.shape[1])
    with col3:
        st.metric("Размер данных (MB)", round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))

def analyze_data_types(df):
    """Анализ типов данных"""
    st.subheader("Типы данных")
    
    # Создаем словарь с предварительно преобразованными типами
    data = {
        'Столбец': list(df.dtypes.index),
        'Тип данных': [str(dtype) for dtype in df.dtypes.values],  # Преобразуем в строки
        'Количество null': list(df.isnull().sum().values),
        'Процент null': list((df.isnull().sum().values / len(df) * 100).round(2))
    }
    
    # Создаем DataFrame с явно указанными типами
    dtypes_df = pd.DataFrame(data).astype({
        'Столбец': str,
        'Тип данных': str,
        'Количество null': int,
        'Процент null': float
    })
    
    st.dataframe(dtypes_df)

def analyze_duplicates(df):
    """
    Анализ дубликатов
    """
    st.subheader("Анализ дубликатов")
    
    duplicates = df.duplicated().sum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Количество дубликатов", duplicates)
    with col2:
        st.metric("Процент дубликатов", round(duplicates / len(df) * 100, 2))

def perform_normality_test(data):
    """
    Оптимизированная версия теста на нормальность
    """
    # Если данных слишком много, берем случайную выборку
    if len(data) > 5000:
        data = data.sample(n=5000, random_state=42)
    
    # Тест Шапиро-Уилка
    if len(data) < 5000:  # Тест работает эффективно на выборках до 5000
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Тест Колмогорова-Смирнова
    ks_stat, ks_p = stats.kstest(stats.zscore(data), 'norm')
    
    # D'Agostino's K^2 тест
    k2_stat, k2_p = stats.normaltest(data)
    
    return {
        'shapiro': (shapiro_stat, shapiro_p) if shapiro_stat is not None else None,
        'ks': (ks_stat, ks_p),
        'k2': (k2_stat, k2_p)
    }

def get_numerical_stats(df):
    """
    Получение статистики по числовым данным
    """
    st.subheader("Статистика числовых данных")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        # Выбор столбца для анализа
        selected_column = st.selectbox(
            "Выберите столбец для подробного анализа",
            numerical_cols
        )
        
        # Проведение расширенного анализа для выбранного столбца
        analyze_distribution(df, selected_column)
        
        # Общая статистика по всем числовым столбцам
        st.subheader("Общая статистика по всем числовым столбцам")
        stats_df = df[numerical_cols].describe()
        st.dataframe(stats_df)
    else:
        st.info("В датасете нет числовых столбцов")

def analyze_outliers(df, column):
    """
    Анализ выбросов в данных с использованием метода IQR
    """
    if column is None:
        st.warning("Выберите столбец для анализа выбросов")
        return None, None
        
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    st.subheader(f"Анализ выбросов: {column}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Межквартильный размах (IQR)", round(IQR, 2))
    with col2:
        st.metric("Количество выбросов", len(outliers))
    with col3:
        st.metric("Процент выбросов", round(len(outliers) / len(df) * 100, 2))
    
    st.write("Границы выбросов:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Нижняя граница", round(lower_bound, 2))
    with col2:
        st.metric("Верхняя граница", round(upper_bound, 2))
    
    # Добавляем описательную статистику выбросов
    if len(outliers) > 0:
        st.write("### Статистика выбросов")
        outliers_stats = outliers.describe()
        st.dataframe(pd.DataFrame({
            'Статистика': outliers_stats.index,
            'Значение': outliers_stats.values
        }))
    
    return lower_bound, upper_bound
