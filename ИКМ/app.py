"""
Веб-интерфейс для прогнозирования спроса на велопрокат.
Фреймворк: Gradio
Запуск: python app.py
"""

import gradio as gr
from predict import predict

# Маппинг человекочитаемых значений в числовые коды
SEASONS = {"Весна": 1, "Лето": 2, "Осень": 3, "Зима": 4}
WEATHER = {
    "Ясно / Малооблачно": 1,
    "Туман / Облачно": 2,
    "Лёгкий дождь / Снег": 3,
    "Сильный дождь / Шторм": 4,
}
WEEKDAYS = {"Вс": 0, "Пн": 1, "Вт": 2, "Ср": 3, "Чт": 4, "Пт": 5, "Сб": 6}


def predict_demand(season, year, month, hour, holiday, weekday,
                   workingday, weather, temp, humidity, windspeed):
    """Обработчик Gradio: преобразует входы и вызывает predict()."""
    # Преобразуем ощущаемую температуру (приблизительно)
    atemp = temp * 0.9 + 0.05

    result = predict(
        season=SEASONS[season],
        yr=int(year == "2012"),
        mnth=month,
        hr=hour,
        holiday=int(holiday),
        weekday=WEEKDAYS[weekday],
        workingday=int(workingday),
        weathersit=WEATHER[weather],
        temp=temp,
        atemp=atemp,
        hum=humidity,
        windspeed=windspeed,
    )

    # Текстовая интерпретация
    result = round(result)
    if result < 50:
        level = "Низкий спрос"
    elif result < 150:
        level = "Средний спрос"
    elif result < 300:
        level = "Высокий спрос"
    else:
        level = "Очень высокий спрос"

    return f"{result} прокатов/час ({level})"


# Создаём интерфейс
demo = gr.Interface(
    fn=predict_demand,
    inputs=[
        gr.Dropdown(list(SEASONS.keys()), label="Сезон", value="Лето"),
        gr.Radio(["2011", "2012"], label="Год", value="2012"),
        gr.Slider(1, 12, step=1, label="Месяц", value=6),
        gr.Slider(0, 23, step=1, label="Час", value=17),
        gr.Checkbox(label="Праздничный день", value=False),
        gr.Dropdown(list(WEEKDAYS.keys()), label="День недели", value="Ср"),
        gr.Checkbox(label="Рабочий день", value=True),
        gr.Dropdown(list(WEATHER.keys()), label="Погода", value="Ясно / Малооблачно"),
        gr.Slider(0, 1, step=0.01, label="Температура (0=холодно, 1=жарко)", value=0.7),
        gr.Slider(0, 1, step=0.01, label="Влажность (0=сухо, 1=влажно)", value=0.5),
        gr.Slider(0, 1, step=0.01, label="Скорость ветра (0=штиль, 1=сильный)", value=0.2),
    ],
    outputs=gr.Textbox(label="Прогноз спроса"),
    title="Прогнозирование спроса на велопрокат",
    description="Модель предсказывает количество велопрокатов за час на основе погоды, времени и дня недели.",
    examples=[
        ["Лето", "2012", 6, 17, False, "Ср", True, "Ясно / Малооблачно", 0.7, 0.5, 0.2],
        ["Зима", "2011", 1, 8, False, "Пн", True, "Туман / Облачно", 0.2, 0.8, 0.3],
        ["Осень", "2012", 10, 12, True, "Сб", False, "Лёгкий дождь / Снег", 0.5, 0.7, 0.4],
    ],
)

if __name__ == "__main__":
    demo.launch()
