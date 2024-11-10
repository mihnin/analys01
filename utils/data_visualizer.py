import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

def analyze_distribution(data):
    """
    Анализ распределения данных
    """
    skewness = stats.skew(data.dropna())
    kurtosis = stats.kurtosis(data.dropna())
    statistic, p_value = stats.normaltest(data.dropna())
    
    distribution_text = ""
    if abs(skewness) < 0.5:
        distribution_text = "близко к нормальному"
    elif skewness > 0:
        distribution_text = "правосторонняя асимметрия"
    else:
        distribution_text = "левосторонняя асимметрия"
        
    return {
        'skewness': round(skewness, 2),
        'kurtosis': round(kurtosis, 2),
        'p_value': p_value,
        'distribution': distribution_text
    }

def create_histogram(df, column):
    """
    Создание улучшенной гистограммы с анализом распределения
    """
    st.subheader(f"Анализ распределения: {column}")
    
    # Анализ распределения
    dist_stats = analyze_distribution(df[column])
    
    # Информация о распределении
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Асимметрия", dist_stats['skewness'])
    with col2:
        st.metric("Эксцесс", dist_stats['kurtosis'])
    with col3:
        st.metric("Тип распределения", dist_stats['distribution'])
    
    # Создание гистограммы с улучшенным оформлением
    fig = px.histogram(
        df, 
        x=column,
        title=f'Распределение значений: {column}',
        template='plotly_white',
        labels={column: f'Значения {column}'},
        hover_data=[column],
        marginal='box'  # Добавляем box plot сверху
    )
    
    # Улучшение оформления
    fig.update_layout(
        showlegend=True,
        xaxis_title=f'{column}',
        yaxis_title='Частота',
        grid={'rows': 1, 'columns': 1},
        plot_bgcolor='white'
    )
    
    # Добавление среднего и медианы
    fig.add_vline(x=df[column].mean(), line_dash="dash", line_color="red",
                 annotation_text="Среднее")
    fig.add_vline(x=df[column].median(), line_dash="dash", line_color="green",
                 annotation_text="Медиана")
    
    st.plotly_chart(fig)

def create_box_plot(df, column):
    """
    Создание улучшенного box plot с описанием статистик
    """
    st.subheader(f"Box Plot анализ: {column}")
    
    # Расчет статистик
    stats_dict = {
        'Медиана': df[column].median(),
        'Q1': df[column].quantile(0.25),
        'Q3': df[column].quantile(0.75),
        'IQR': df[column].quantile(0.75) - df[column].quantile(0.25),
        'Минимум': df[column].min(),
        'Максимум': df[column].max()
    }
    
    # Вывод статистик
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Медиана", round(stats_dict['Медиана'], 2))
    with col2:
        st.metric("IQR", round(stats_dict['IQR'], 2))
    with col3:
        st.metric("Размах", round(stats_dict['Максимум'] - stats_dict['Минимум'], 2))
    
    # Создание box plot с улучшенным оформлением
    fig = px.box(
        df, 
        y=column,
        title=f'Box Plot: {column}',
        template='plotly_white',
        points='all',  # Показывать все точки
        labels={column: f'Значения {column}'}
    )
    
    # Улучшение оформления
    fig.update_layout(
        showlegend=True,
        yaxis_title=f'{column}',
        grid={'rows': 1, 'columns': 1},
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig)

def create_scatter_plot(df, x_column, y_column):
    """
    Создание улучшенного scatter plot с анализом корреляции
    """
    st.subheader(f"Анализ взаимосвязи: {x_column} vs {y_column}")
    
    # Расчет корреляции
    correlation = df[x_column].corr(df[y_column])
    correlation_text = ""
    if abs(correlation) < 0.3:
        correlation_text = "слабая"
    elif abs(correlation) < 0.7:
        correlation_text = "средняя"
    else:
        correlation_text = "сильная"
    
    # Вывод информации о корреляции
    st.metric("Корреляция", f"{round(correlation, 3)} ({correlation_text})")
    
    # Создание scatter plot с улучшенным оформлением
    fig = px.scatter(
        df, 
        x=x_column, 
        y=y_column,
        title=f'Диаграмма рассеяния: {x_column} vs {y_column}',
        template='plotly_white',
        trendline="ols",  # Добавление линии тренда
        labels={
            x_column: f'Значения {x_column}',
            y_column: f'Значения {y_column}'
        },
        hover_data=[x_column, y_column]
    )
    
    # Улучшение оформления
    fig.update_layout(
        showlegend=True,
        grid={'rows': 1, 'columns': 1},
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig)

def plot_correlation_matrix(df):
    """
    Построение улучшенной корреляционной матрицы
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr()
        
        # Создание тепловой карты корреляций
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Корреляция"),
            title="Корреляционная матрица",
            template='plotly_white',
            color_continuous_scale='RdBu',  # Красно-синяя цветовая схема
            aspect='auto'
        )
        
        # Улучшение оформления
        fig.update_layout(
            plot_bgcolor='white',
            width=800,
            height=800
        )
        
        st.plotly_chart(fig)
        
        # Выделение сильных корреляций
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i,j]) > 0.7:
                    strong_correlations.append({
                        'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        'correlation': round(corr_matrix.iloc[i,j], 3)
                    })
        
        if strong_correlations:
            st.subheader("Сильные корреляции (|r| > 0.7):")
            for corr in strong_correlations:
                st.write(f"{corr['pair']}: {corr['correlation']}")
    else:
        st.info("Недостаточно числовых столбцов для построения корреляционной матрицы")

def plot_missing_values(df):
    """
    Визуализация пропущенных значений
    """
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        # Создание столбчатой диаграммы
        fig = px.bar(
            x=missing_data.index, 
            y=missing_data.values,
            title="Количество пропущенных значений по столбцам",
            template='plotly_white',
            labels={
                'x': 'Столбец',
                'y': 'Количество пропущенных значений'
            }
        )
        
        # Улучшение оформления
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            grid={'rows': 1, 'columns': 1}
        )
        
        st.plotly_chart(fig)
        
        # Добавление процентного соотношения
        st.write("Процент пропущенных значений:")
        missing_percent = (missing_data / len(df) * 100).round(2)
        for col, percent in missing_percent[missing_percent > 0].items():
            st.write(f"{col}: {percent}%")
    else:
        st.info("В датасете нет пропущенных значений")

def plot_outliers(df, column, lower_bound, upper_bound):
    """
    Визуализация выбросов с улучшенным оформлением
    """
    st.subheader(f"Анализ выбросов: {column}")
    
    # Box plot с выбросами
    fig_box = px.box(
        df, 
        y=column,
        title=f'Box Plot с выбросами: {column}',
        template='plotly_white',
        points='outliers'  # Показывать только выбросы
    )
    
    # Улучшение оформления box plot
    fig_box.update_layout(
        showlegend=True,
        yaxis_title=f'Значения {column}',
        plot_bgcolor='white',
        grid={'rows': 1, 'columns': 1}
    )
    
    st.plotly_chart(fig_box)
    
    # Гистограмма с границами выбросов
    fig_hist = px.histogram(
        df, 
        x=column,
        title=f'Распределение значений с границами выбросов: {column}',
        template='plotly_white',
        labels={column: f'Значения {column}'}
    )
    
    # Добавление границ выбросов
    fig_hist.add_vline(x=lower_bound, line_dash="dash", line_color="red",
                      annotation_text="Нижняя граница")
    fig_hist.add_vline(x=upper_bound, line_dash="dash", line_color="red",
                      annotation_text="Верхняя граница")
    
    # Улучшение оформления гистограммы
    fig_hist.update_layout(
        showlegend=True,
        xaxis_title=f'{column}',
        yaxis_title='Частота',
        plot_bgcolor='white',
        grid={'rows': 1, 'columns': 1}
    )
    
    st.plotly_chart(fig_hist)
