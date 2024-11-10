import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import streamlit as st

def prepare_data(df, target_column, feature_columns, task_type='regression'):
    """
    Подготовка данных для обучения модели с улучшенной валидацией и обработкой ошибок
    """
    try:
        # Проверка корректности task_type
        if task_type not in ['regression', 'classification']:
            st.error("❌ Ошибка: task_type должен быть 'regression' или 'classification'")
            return None, None, None, None, None

        # Проверка типа целевой переменной
        if task_type == 'regression' and not pd.api.types.is_numeric_dtype(df[target_column]):
            st.error("❌ Для регрессии требуется числовая целевая переменная")
            return None, None, None, None, None
        
        if task_type == 'classification' and pd.api.types.is_numeric_dtype(df[target_column]):
            st.error("❌ Для классификации требуется категориальная целевая переменная")
            return None, None, None, None, None

        # Проверка на пустые значения в целевой переменной
        target_nulls = df[target_column].isnull().sum()
        if target_nulls > 0:
            st.warning(f"⚠️ Обнаружено {target_nulls} пропущенных значений в целевой переменной. Удаляем строки...")
            df = df.dropna(subset=[target_column])
            if len(df) < 10:
                st.error("❌ После удаления пропусков осталось слишком мало данных для обучения")
                return None, None, None, None, None

        # Обработка пропусков в признаках
        for column in feature_columns:
            nulls = df[column].isnull().sum()
            if nulls > 0:
                if pd.api.types.is_numeric_dtype(df[column]):
                    st.info(f"ℹ️ Заполняем {nulls} пропусков медианой в столбце '{column}'")
                    df[column] = df[column].fillna(df[column].median())
                else:
                    st.info(f"ℹ️ Заполняем {nulls} пропусков модой в столбце '{column}'")
                    df[column] = df[column].fillna(df[column].mode()[0])

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Проверка на оставшиеся пропуски
        if X.isnull().any().any() or y.isnull().any():
            st.error("❌ После обработки остались пропущенные значения")
            return None, None, None, None, None

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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if task_type == 'classification' else None
        )

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
    Обучение модели с улучшенной валидацией и обработкой ошибок
    """
    try:
        # Проверка входных данных
        if X_train is None or y_train is None:
            st.error("❌ Отсутствуют данные для обучения")
            return None

        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            st.error("❌ Неверный формат входных данных")
            return None

        # Проверка размерности данных
        if len(X_train) != len(y_train):
            st.error("❌ Размерности X_train и y_train не совпадают")
            return None

        if len(X_train) < 10:
            st.error("❌ Недостаточно данных для обучения (минимум 10 образцов)")
            return None

        # Валидация task_type
        if task_type not in ['regression', 'classification']:
            st.error("❌ Неверный тип задачи. Используйте 'regression' или 'classification'")
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

        # Обучение модели с обработкой возможных ошибок
        try:
            model.fit(X_train, y_train)
        except ValueError as ve:
            st.error(f"❌ Ошибка при обучении модели: {str(ve)}")
            return None
        except Exception as e:
            st.error(f"❌ Неожиданная ошибка при обучении: {str(e)}")
            return None

        # Проверка качества модели на кросс-валидации
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            if task_type == 'regression':
                st.info(f"ℹ️ R² на кросс-валидации: {mean_score:.3f} ± {std_score*2:.3f}")
            else:
                st.info(f"ℹ️ Accuracy на кросс-валидации: {mean_score:.3f} ± {std_score*2:.3f}")

            if mean_score < 0.1:  # Базовый порог качества
                st.warning("⚠️ Модель показывает очень низкое качество на кросс-валидации")

        except Exception as e:
            st.warning(f"⚠️ Не удалось выполнить кросс-валидацию: {str(e)}")

        return model

    except Exception as e:
        st.error(f"❌ Ошибка при создании модели: {str(e)}")
        return None

def evaluate_model(model, X_test, y_test, task_type='regression'):
    """
    Оценка качества модели с улучшенной обработкой ошибок
    """
    try:
        # Проверка входных данных
        if model is None or X_test is None or y_test is None:
            st.error("❌ Отсутствуют данные для оценки модели")
            return None

        predictions = model.predict(X_test)
        
        if task_type == 'regression':
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # Проверка качества регрессии
            if r2 < 0:
                st.warning("⚠️ Модель работает хуже, чем простое среднее значение (R² < 0)")
            
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
            
            # Проверка качества классификации
            if accuracy < 1.0 / len(np.unique(y_test)):
                st.warning("⚠️ Модель работает хуже случайного угадывания")
            
            return metrics

    except Exception as e:
        st.error(f"❌ Ошибка при оценке модели: {str(e)}")
        return None

def plot_feature_importance(model, feature_names):
    """
    Визуализация важности признаков с улучшенной обработкой ошибок
    """
    try:
        # Проверка входных данных
        if model is None or feature_names is None:
            st.error("❌ Отсутствуют данные для визуализации важности признаков")
            return None

        if not hasattr(model, 'feature_importances_'):
            st.error("❌ Модель не поддерживает анализ важности признаков")
            return None

        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        if importance['importance'].isnull().any():
            st.error("❌ Обнаружены некорректные значения важности признаков")
            return None

        importance = importance.sort_values('importance', ascending=False)
        importance['cumulative_importance'] = importance['importance'].cumsum()
        
        # Выводим топ признаков
        st.write("### Топ важных признаков:")
        for idx, row in importance.head().iterrows():
            st.write(f"- {row['feature']}: {row['importance']:.3f} "
                    f"(Накопленная важность: {row['cumulative_importance']:.3f})")
        
        # Визуализация
        st.bar_chart(importance.set_index('feature'))
        
        # Анализ важности признаков
        if importance['importance'].max() > 0.8:
            st.warning("⚠️ Возможна сильная зависимость от одного признака")
        
        return importance

    except Exception as e:
        st.error(f"❌ Ошибка при визуализации важности признаков: {str(e)}")
        return None

def plot_predictions(y_test, predictions, task_type='regression', encoders=None):
    """
    Визуализация результатов прогнозирования с улучшенной обработкой ошибок
    """
    try:
        # Проверка входных данных
        if y_test is None or predictions is None:
            st.error("❌ Отсутствуют данные для визуализации")
            return

        if len(y_test) != len(predictions):
            st.error("❌ Размерности y_test и predictions не совпадают")
            return

        if task_type == 'regression':
            results_df = pd.DataFrame({
                'Фактические значения': y_test,
                'Прогноз': predictions
            })
            
            # График сравнения
            st.line_chart(results_df)
            
            # Scatter plot
            import plotly.express as px
            fig = px.scatter(
                x=y_test,
                y=predictions,
                labels={'x': 'Фактические значения', 'y': 'Прогноз'},
                title='Сравнение прогнозов с фактическими значениями'
            )
            
            # Добавление линии идеального предсказания
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
            
            # Получение меток классов
            if encoders and 'target' in encoders:
                labels = encoders['target'].classes_
            else:
                labels = [str(i) for i in range(len(np.unique(y_test)))]
            
            # Создание тепловой карты
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

            # Анализ несбалансированности классов
            class_counts = np.bincount(y_test)
            if max(class_counts) / min(class_counts) > 5:
                st.warning("⚠️ Обнаружен существенный дисбаланс классов")

    except Exception as e:
        st.error(f"❌ Ошибка при визуализации результатов: {str(e)}")
