import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import streamlit as st

def prepare_data(df, target_column, feature_columns):
    """
    Подготовка данных для обучения модели
    """
    X = df[feature_columns]
    y = df[target_column]
    
    # Обработка категориальных признаков
    X = pd.get_dummies(X, drop_first=True)
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Стандартизация числовых признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train, task_type='regression'):
    """
    Обучение модели в зависимости от типа задачи
    """
    if task_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, task_type='regression'):
    """
    Оценка качества модели
    """
    predictions = model.predict(X_test)
    
    if task_type == 'regression':
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        return {
            'RMSE': round(rmse, 3),
            'R2': round(r2, 3)
        }
    else:
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            'Accuracy': round(accuracy, 3),
            'Report': report
        }

def plot_feature_importance(model, feature_names):
    """
    Визуализация важности признаков
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    st.bar_chart(importance.set_index('feature'))

def plot_predictions(y_test, predictions, task_type='regression'):
    """
    Визуализация результатов прогнозирования
    """
    if task_type == 'regression':
        results_df = pd.DataFrame({
            'Фактические значения': y_test,
            'Прогноз': predictions
        })
        
        st.line_chart(results_df)
    else:
        # Для классификации показываем матрицу ошибок
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff
        
        cm = confusion_matrix(y_test, predictions)
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=['Predicted ' + str(x) for x in range(len(np.unique(y_test)))],
            y=['Actual ' + str(x) for x in range(len(np.unique(y_test)))],
            colorscale='Viridis'
        )
        st.plotly_chart(fig)
