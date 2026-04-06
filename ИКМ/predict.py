"""
Модуль предсказания: загружает обученную модель и делает прогноз.
Используется в app.py (Gradio) и tests/test_basic.py.
"""

import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "best_model.pkl"
_model = None


def load_model():
    """Загружает модель из .pkl файла (ленивая загрузка — один раз)."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(season, yr, mnth, hr, holiday, weekday, workingday,
            weathersit, temp, atemp, hum, windspeed):
    """
    Прогнозирует количество велопрокатов за час.

    Параметры:
        season      — сезон (1=весна, 2=лето, 3=осень, 4=зима)
        yr          — год (0=2011, 1=2012)
        mnth        — месяц (1-12)
        hr          — час (0-23)
        holiday     — праздник (0/1)
        weekday     — день недели (0=Вс, 1=Пн, ..., 6=Сб)
        workingday  — рабочий день (0/1)
        weathersit  — погода (1=ясно, 2=туман, 3=дождь, 4=шторм)
        temp        — температура (0-1, нормализованная)
        atemp       — ощущаемая температура (0-1)
        hum         — влажность (0-1)
        windspeed   — скорость ветра (0-1)

    Возвращает:
        float — прогноз количества прокатов
    """
    model = load_model()

    features = pd.DataFrame([{
        "season": int(season),
        "yr": int(yr),
        "mnth": int(mnth),
        "hr": int(hr),
        "holiday": int(holiday),
        "weekday": int(weekday),
        "workingday": int(workingday),
        "weathersit": int(weathersit),
        "temp": float(temp),
        "atemp": float(atemp),
        "hum": float(hum),
        "windspeed": float(windspeed),
    }])

    result = model.predict(features)[0]
    return max(0.0, float(result))  # прокатов не может быть меньше 0
