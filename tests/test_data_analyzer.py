import unittest
import numpy as np
import pandas as pd
from your_module import perform_normality_test, analyze_distribution  # Замените 'your_module' на имя вашего модуля

class TestDataAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Создаем тестовый датафрейм
        np.random.seed(42)
        cls.test_df = pd.DataFrame({
            'normal': np.random.normal(0, 1, 1000),
            'uniform': np.random.uniform(0, 1, 1000),
            'categorical': np.random.choice(['A', 'B', 'C'], 1000),
            'nulls': [None] * 100 + list(range(900))
        })

    def test_perform_normality_test(self):
        # Тест для нормального распределения
        result = perform_normality_test(self.test_df['normal'])
        self.assertIsNotNone(result['shapiro'])
        self.assertIsNotNone(result['ks'])
        self.assertIsNotNone(result['k2'])

    def test_analyze_distribution(self):
        # Тест анализа распределения
        stats_dict = analyze_distribution(self.test_df, 'normal')
        self.assertIsNotNone(stats_dict)
        self.assertTrue('Среднее' in stats_dict)
        self.assertTrue('Медиана' in stats_dict)

if __name__ == '__main__':
    unittest.main()
