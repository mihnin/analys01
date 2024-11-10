import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Создаем даты
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='M')

# Задаем регионы и категории продуктов
regions = ['Центр', 'Север', 'Юг', 'Восток']
product_categories = ['Электроника', 'Одежда', 'Продукты']

# Создаем базовые данные
np.random.seed(42)
n_records = len(dates) * len(regions) * len(product_categories)

data = {
    'date': np.repeat(dates, len(regions) * len(product_categories)),
    'region': np.tile(np.repeat(regions, len(product_categories)), len(dates)),
    'product_category': np.tile(product_categories, len(dates) * len(regions))
}

# Генерируем числовые данные с трендами
df = pd.DataFrame(data)

# Создаем базовые значения с трендом
base_sales = np.linspace(100, 200, len(dates))
base_customers = np.linspace(50, 150, len(dates))

# Добавляем сезонность и случайность
df['sales'] = np.tile(base_sales, len(regions) * len(product_categories)) * \
              (1 + 0.3 * np.sin(np.linspace(0, 4*np.pi, len(df)))) + \
              np.random.normal(0, 20, len(df))

df['customers'] = np.tile(base_customers, len(regions) * len(product_categories)) * \
                 (1 + 0.2 * np.cos(np.linspace(0, 4*np.pi, len(df)))) + \
                 np.random.normal(0, 10, len(df))

# Округляем customers до целых чисел
df['customers'] = df['customers'].round().astype(int)

# Генерируем выручку и маржинальность
df['revenue'] = df['sales'] * np.random.uniform(100, 150, len(df))
df['profit_margin'] = np.random.uniform(0.15, 0.35, len(df))

# Добавляем пропущенные значения (5% от всех значений)
for column in ['sales', 'customers', 'revenue', 'profit_margin']:
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, column] = np.nan

# Добавляем дубликаты (2% от всех записей)
n_duplicates = int(len(df) * 0.02)
duplicate_indices = np.random.choice(len(df), n_duplicates, replace=False)
duplicates = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# Преобразуем категориальные столбцы
df['region'] = df['region'].astype('category')
df['product_category'] = df['product_category'].astype('category')

# Сортируем по дате
df = df.sort_values('date').reset_index(drop=True)

# Сохраняем в CSV
df.to_csv('test_data.csv', index=False)

print("Тестовый набор данных создан и сохранен в test_data.csv")
print(f"Размер датасета: {len(df)} строк")
print("\nПримеры данных:")
print(df.head())
print("\nОписательная статистика:")
print(df.describe())
print("\nИнформация о датасете:")
print(df.info())
