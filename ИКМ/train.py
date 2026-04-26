"""
Обучение модели прогнозирования спроса на велопрокат.
Датасет: UCI Bike Sharing Dataset (hour.csv)
Задача: регрессия — предсказать количество прокатов (cnt)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from pathlib import Path

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────────────────────────
# СОБСТВЕННАЯ РЕАЛИЗАЦИЯ ПРОСТОЙ МОДЕЛИ (baseline)
# ─────────────────────────────────────────────────────────────────
# Простое правило на одном признаке: предсказываем среднее количество
# прокатов за конкретный час суток. Игнорируем все остальные признаки.
# Это базовая модель — сложные алгоритмы должны работать лучше неё.

class HourlyMeanRegressor(BaseEstimator, RegressorMixin):
    """Базовая модель: предсказание = среднее количество прокатов за этот час."""

    def fit(self, X, y):
        # Считаем среднее значение y для каждого часа (0-23)
        df = pd.DataFrame({"hr": X["hr"].values, "y": y.values if hasattr(y, "values") else y})
        self.hour_means_ = df.groupby("hr")["y"].mean().to_dict()
        self.global_mean_ = df["y"].mean()  # запасной вариант
        return self

    def predict(self, X):
        # Для каждой строки берём среднее за её час
        return np.array([self.hour_means_.get(h, self.global_mean_) for h in X["hr"].values])

# ─────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "bike_data" / "hour.csv")

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"Пропуски: {df.isnull().sum().sum()}")

# Удаляем столбцы, которые нельзя использовать как признаки:
#   instant — просто индекс строки
#   dteday  — дата в строковом формате (информация уже в yr, mnth, hr, weekday)
#   casual + registered = cnt — это утечка данных (data leakage)
df = df.drop(columns=["instant", "dteday", "casual", "registered"])

# Целевая переменная — cnt (общее количество прокатов за час)
target = "cnt"
y = df[target]
X = df.drop(columns=[target])

print(f"\nПризнаки ({X.shape[1]}): {list(X.columns)}")
print(f"Целевая переменная: {target}")
print(f"  Среднее: {y.mean():.0f} прокатов/час")
print(f"  Медиана: {y.median():.0f}")
print(f"  Мин: {y.min()}, Макс: {y.max()}")

# Категориальные признаки (закодированы числами, но по смыслу — категории)
cat_features = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
# Числовые признаки (непрерывные, нормализованные в датасете)
num_features = ["temp", "atemp", "hum", "windspeed"]

print(f"\nКатегориальные ({len(cat_features)}): {cat_features}")
print(f"Числовые ({len(num_features)}): {num_features}")

# ─────────────────────────────────────────────────────────────────
# Предобработка: ColumnTransformer
# ─────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
    ("num", StandardScaler(), num_features),
])

# Разделение: 80% обучение, 20% тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка:  {X_test.shape[0]} строк")

# ─────────────────────────────────────────────────────────────────
# 2. ОБУЧЕНИЕ И КРОСС-ВАЛИДАЦИЯ ДВУХ МОДЕЛЕЙ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. ОБУЧЕНИЕ И КРОСС-ВАЛИДАЦИЯ")
print("=" * 60)

# Принципиально разные подходы:
#   1. Baseline (своя реализация) — простое правило на одном признаке (среднее за час)
#   2. RandomForest — ансамбль независимых деревьев (bagging)
#   3. GradientBoosting — последовательный ансамбль (boosting)
# Baseline нужен, чтобы понимать "стартовую планку" — насколько лучше работают сложные модели
pipelines = {
    "Baseline (среднее за час)": HourlyMeanRegressor(),
    "RandomForest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    "GradientBoosting": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=5))
    ]),
}

cv_results = {}

for name, pipe in pipelines.items():
    print(f"\n--- {name} ---")

    # 5-fold кросс-валидация по MAE
    cv_mae = -cross_val_score(pipe, X_train, y_train, cv=5,
                               scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(pipe, X_train, y_train, cv=5,
                                        scoring="neg_mean_squared_error", n_jobs=-1))

    print(f"  CV MAE:  {cv_mae.mean():.2f} +/- {cv_mae.std():.2f}")
    print(f"  CV RMSE: {cv_rmse.mean():.2f} +/- {cv_rmse.std():.2f}")

    cv_results[name] = {
        "cv_mae_mean": cv_mae.mean(),
        "cv_mae_std": cv_mae.std(),
        "cv_rmse_mean": cv_rmse.mean(),
    }

# ─────────────────────────────────────────────────────────────────
# 3. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 60)

test_results = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    test_results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}

    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.4f}")

# Определяем лучшую модель
best_name = min(test_results, key=lambda k: test_results[k]["mae"])
best_pipe = pipelines[best_name]

print(f"\nЛучшая модель: {best_name} (MAE = {test_results[best_name]['mae']:.2f})")

# ─────────────────────────────────────────────────────────────────
# 4. ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ВИЗУАЛИЗАЦИЯ")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4.1 Scatter: предсказания vs реальность (лучшая модель)
y_pred_best = test_results[best_name]["y_pred"]
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.3, s=10)
axes[0, 0].plot([0, y_test.max()], [0, y_test.max()], "r--", linewidth=2)
axes[0, 0].set_xlabel("Реальное количество прокатов")
axes[0, 0].set_ylabel("Предсказанное количество")
axes[0, 0].set_title(f"Предсказания: {best_name}\nR2={test_results[best_name]['r2']:.4f}")
axes[0, 0].grid(True, alpha=0.3)

# 4.2 Распределение ошибок (остатки)
residuals = y_test.values - y_pred_best
axes[0, 1].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
axes[0, 1].axvline(x=0, color="red", linestyle="--", linewidth=2)
axes[0, 1].set_xlabel("Ошибка (реальное - предсказанное)")
axes[0, 1].set_ylabel("Частота")
axes[0, 1].set_title("Распределение ошибок")
axes[0, 1].grid(True, alpha=0.3)

# 4.3 Сравнение MAE двух моделей
model_names = list(test_results.keys())
mae_values = [test_results[n]["mae"] for n in model_names]
colors = ["#2ecc71" if n == best_name else "#3498db" for n in model_names]
bars = axes[1, 0].bar(model_names, mae_values, color=colors, edgecolor="white")
axes[1, 0].set_ylabel("MAE (прокатов)")
axes[1, 0].set_title("Сравнение моделей по MAE")
for bar, val in zip(bars, mae_values):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", fontweight="bold")

# 4.4 Важность признаков (для лучшей модели)
model_step = best_pipe.named_steps["model"]
prep_step = best_pipe.named_steps["prep"]

if hasattr(model_step, "feature_importances_"):
    # Восстанавливаем имена признаков после ColumnTransformer
    cat_encoder = prep_step.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_names = cat_names + num_features

    importances = model_step.feature_importances_
    top_n = 15
    idx = np.argsort(importances)[::-1][:top_n]

    axes[1, 1].barh(
        [all_names[i] for i in reversed(idx)],
        importances[idx[::-1]],
        color="#9b59b6"
    )
    axes[1, 1].set_xlabel("Важность")
    axes[1, 1].set_title(f"Топ-{top_n} признаков ({best_name})")

plt.tight_layout()
plt.savefig(BASE_DIR / "training_results.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────
# 5. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. СОХРАНЕНИЕ МОДЕЛИ")
print("=" * 60)

model_path = BASE_DIR / "models" / "best_model.pkl"
os.makedirs(model_path.parent, exist_ok=True)
joblib.dump(best_pipe, model_path)
print(f"Модель сохранена: {model_path}")

# ─────────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ОТЧЁТ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ОТЧЁТ")
print("=" * 60)

# Анализ ошибок: где модель ошибается больше всего
y_pred_all = best_pipe.predict(X_test)
errors = np.abs(y_test.values - y_pred_all)
error_by_hour = pd.DataFrame({"hr": X_test["hr"].values, "error": errors})
worst_hours = error_by_hour.groupby("hr")["error"].mean().nlargest(3)
worst_hours_str = ", ".join(f"{hr}:00" for hr in worst_hours.index.tolist())

# Краткий отчёт в формате из задания:
print(f"\n  Лучшая модель — {best_name}.")
print(f"  Её ключевая метрика на новых данных — MAE = {test_results[best_name]['mae']:.2f} прокатов")
print(f"  (RMSE = {test_results[best_name]['rmse']:.2f}, R2 = {test_results[best_name]['r2']:.4f}).")
print(f"  Чаще всего она ошибается в часы пик: {worst_hours_str}")
print(f"  → Модель объясняет {test_results[best_name]['r2'] * 100:.1f}% дисперсии спроса")

print(f"\n  Часы с наибольшими ошибками (детально):")
for hr, err in worst_hours.items():
    print(f"    {hr}:00 — средняя ошибка {err:.0f} прокатов")

print(f"\nФайл модели: {model_path}")
print("=" * 60)
