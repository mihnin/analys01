import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import streamlit as st

def prepare_data(df, target_column, feature_columns, task_type='regression'):
    """
    Подготовка данных для обучения модели
    """
    try:
        # Проверка типа целевой переменной
        is_target_numeric = np.issubdtype(df[target_column].dtype, np.number)
        if task_type == 'regression' and not is_target_numeric:
            st.error(f"❌ Ошибка: Для задачи регрессии целевая переменная '{target_column}' должна быть числовой.")
            return None, None, None, None, None

        elif task_type == 'classification' and is_target_numeric:
            st.error(f"❌ Ошибка: Для задачи классификации целевая переменная '{target_column}' должна быть категориальной.")
            return None, None, None, None, None

        # Обработка пропущенных значений
        if df[target_column].isnull().any():
            st.warning(f"⚠️ Удаление строк с пропущенными значениями в целевой переменной '{target_column}'")
            df = df.dropna(subset=[target_column])

        # Обработка пропусков в признаках
        for column in feature_columns:
            if df[column].isnull().any():
                if np.issubdtype(df[column].dtype, np.number):
                    st.info(f"ℹ️ Заполнение пропусков медианой в столбце '{column}'")
                    df[column] = df[column].fillna(df[column].median())
                else:
                    st.info(f"ℹ️ Заполнение пропусков модой в столбце '{column}'")
                    df[column] = df[column].fillna(df[column].mode()[0])

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Кодирование категориальных признаков
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        label_encoders = {}
        
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column].astype(str))

        # Кодирование целевой переменной для классификации
        if task_type == 'classification':
            label_encoders['target'] = LabelEncoder()
            y = label_encoders['target'].fit_transform(y.astype(str))

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Стандартизация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, (scaler, label_encoders)

    except Exception as e:
        st.error(f"❌ Ошибка при подготовке данных: {str(e)}")
        return None, None, None, None, None

def train_model(X_train, y_train, task_type='regression'):
    """
    Обучение модели
    """
    try:
        # Проверка входных данных
        if X_train is None or y_train is None:
            st.error("❌ Ошибка: Отсутствуют данные для обучения")
            return None

        # Настройка параметров модели
        if task_type == 'regression':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )

        # Обучение модели
        model.fit(X_train, y_train)

        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        st.info(f"ℹ️ Средняя оценка кросс-валидации: {cv_scores.mean():.3f} ± {cv_scores.std()*2:.3f}")

        return model

    except Exception as e:
        st.error(f"❌ Ошибка при обучении модели: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, task_type='regression'):
    """
    Оценка качества модели
    """
    try:
        predictions = model.predict(X_test)
        
        if task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            return {
                'RMSE': round(rmse, 3),
                'R2': round(r2, 3),
                'MSE': round(mse, 3)
            }
        else:
            accuracy = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions, output_dict=True)
            
            metrics = {
                'Accuracy': round(accuracy, 3),
                'Report': report
            }
            
            # Добавляем метрики для каждого класса
            for class_label in np.unique(y_test):
                metrics[f'Class_{class_label}_F1'] = round(
                    report[str(class_label)]['f1-score'], 3
                )
            
            return metrics

    except Exception as e:
        st.error(f"❌ Ошибка при оценке модели: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """
    Визуализация важности признаков с дополнительной информацией
    """
    try:
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        # Добавляем информацию о кумулятивной важности
        importance['cumulative_importance'] = importance['importance'].cumsum()
        
        st.write("### Топ 5 важных признаков:")
        for idx, row in importance.head().iterrows():
            st.write(f"- {row['feature']}: {row['importance']:.3f} "
                    f"(Накопленная важность: {row['cumulative_importance']:.3f})")
        
        st.bar_chart(importance.set_index('feature'))
        
        return importance

    except Exception as e:
        st.error(f"❌ Ошибка при визуализации важности признаков: {str(e)}")
        return None

def plot_predictions(y_test, predictions, task_type='regression', encoders=None):
    """
    Визуализация результатов прогнозирования
    """
    try:
        if task_type == 'regression':
            results_df = pd.DataFrame({
                'Фактические значения': y_test,
                'Прогноз': predictions
            })
            
            st.line_chart(results_df)
            
            # Добавляем scatter plot
            import plotly.express as px
            fig = px.scatter(
                x=y_test,
                y=predictions,
                labels={'x': 'Фактические значения', 'y': 'Прогноз'},
                title='Сравнение прогнозов с фактическими значениями'
            )
            fig.add_shape(
                type='line',
                x0=min(y_test),
                y0=min(y_test),
                x1=max(y_test),
                y1=max(y_test),
                line=dict(color='red', dash='dash')
            )
            st.plotly_chart(fig)
            
        else:
            # Матрица ошибок
            from sklearn.metrics import confusion_matrix
            import plotly.figure_factory as ff
            
            cm = confusion_matrix(y_test, predictions)
            
            # Если есть энкодер для целевой переменной, используем оригинальные метки
            if encoders and 'target' in encoders:
                labels = encoders['target'].classes_
            else:
                labels = [str(i) for i in range(len(np.unique(y_test)))]
            
            fig = ff.create_annotated_heatmap(
                z=cm,
                x=labels,
                y=labels,
                annotation_text=cm,
                colorscale='Viridis'
            )
            fig.update_layout(
                title='Матрица ошибок',
                xaxis_title='Прогноз',
                yaxis_title='Фактические значения'
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Ошибка при визуализации результатов: {str(e)}")
