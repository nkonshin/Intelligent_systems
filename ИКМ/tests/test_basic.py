"""
Тесты для проекта прогнозирования спроса на велопрокат.
Запуск: pytest tests/ -v
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_predict_loads_and_works():
    """Тест 1: Функция предсказания загружается и работает на корректном примере."""
    from predict import predict

    result = predict(
        season=1, yr=0, mnth=1, hr=10,
        holiday=0, weekday=1, workingday=1,
        weathersit=1, temp=0.5, atemp=0.5,
        hum=0.5, windspeed=0.2
    )
    assert result is not None, "Функция вернула None"


def test_predict_returns_correct_format():
    """Тест 2: Предсказание возвращает число в правильном формате."""
    from predict import predict

    result = predict(
        season=2, yr=1, mnth=6, hr=17,
        holiday=0, weekday=3, workingday=1,
        weathersit=1, temp=0.7, atemp=0.65,
        hum=0.4, windspeed=0.15
    )
    assert isinstance(result, (int, float)), f"Ожидалось число, получено {type(result)}"
    assert result >= 0, f"Количество прокатов не может быть отрицательным: {result}"


def test_app_creates_without_errors():
    """Тест 3: Веб-приложение (Gradio) создаётся без ошибок."""
    import app

    assert hasattr(app, "demo"), "В app.py не найден объект 'demo'"
