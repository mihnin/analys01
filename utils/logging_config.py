import logging
import logging.handlers
import os
from datetime import datetime
import sys

def setup_logging():
    """Централизованная настройка логирования"""
    try:
        # Создаем директорию для логов
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Исправляем форматтер для корректной работы с UTF-8
        detailed_formatter = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d - %(levelоvel)s - %(name)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Файловый handler с корректной кодировкой
        log_file = os.path.join(log_dir, 'app.log')
        time_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8',
            delay=False
        )
        time_handler.setFormatter(detailed_formatter)
        time_handler.setLevel(logging.DEBUG)
        
        # Очистка всех существующих handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Установка нового handler
        root_logger.addHandler(time_handler)
        root_logger.setLevel(logging.DEBUG)
        
        # Проверка работы логирования
        root_logger.info("="*50)
        root_logger.info("Логирование инициализировано успешно")
        root_logger.info("Тестовая русская строка")
        root_logger.info("="*50)
        
        return root_logger
        
    except Exception as e:
        print(f"Ошибка при настройке логирования: {str(e)}")
        raise
