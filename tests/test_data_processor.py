import unittest
import pandas as pd
import numpy as np
from utils.data_processor import (change_column_type, handle_missing_values,
                                remove_duplicates)

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Создаем тестовый датафрейм
        cls.test_df = pd.DataFrame({
            'numbers': [1, 2, 2, None, 4, 5],
            'text': ['a', 'b', 'b', None, 'd', 'e']
        })

    def test_change_column_type(self):
        df, success = change_column_type(self.test_df.copy(), 'numbers', 'float64')
        self.assertTrue(success)
        self.assertEqual(df['numbers'].dtype, np.float64)

    def test_handle_missing_values(self):
        df, success = handle_missing_values(self.test_df.copy(), 'numbers', 'fill_mean')
        self.assertTrue(success)
        self.assertTrue(df['numbers'].isnull().sum() == 0)

    def test_remove_duplicates(self):
        df, success = remove_duplicates(self.test_df.copy())
        self.assertTrue(success)
        self.assertEqual(len(df), 5)  # Должно быть на 1 строку меньше

if __name__ == '__main__':
    unittest.main()
