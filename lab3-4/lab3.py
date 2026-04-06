import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)

# ─────────────────────────────────────────────────────────────────
# 1. ИССЛЕДОВАНИЕ ДАННЫХ И ПОДГОТОВКА
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("РАЗДЕЛ 1: ИССЛЕДОВАНИЕ ДАННЫХ И ПОДГОТОВКА")
print("=" * 60)

# Загрузка датасета
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
target_names = wine.target_names

print(f"Количество классов: {len(np.unique(y))} ({target_names})")
print(f"Количество признаков: {X.shape[1]}")
print(f"Размер датасета: {X.shape[0]} образцов")

print("\nРаспределение классов:")
class_dist = pd.Series(y).value_counts().sort_index()
for i, count in class_dist.items():
    print(f"  Класс {i} ({target_names[i]}): {count} образцов ({count / len(y) * 100:.1f}%)")

# Проверка дисбаланса
if class_dist.min() / class_dist.max() < 0.7:
    print("\n⚠ Наблюдается небольшой дисбаланс классов (минорный класс составляет "
          f"{class_dist.min() / class_dist.max() * 100:.1f}% от мажорного)")
else:
    print("\n✓ Дисбаланс классов незначительный")

# Разделение со стратификацией (сохраняем пропорции классов)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка:  {X_test.shape[0]} строк")
print("Стратификация применена для сохранения пропорций классов")

# ─────────────────────────────────────────────────────────────────
# 2. ОБУЧЕНИЕ И БАЗОВАЯ ОЦЕНКА
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("РАЗДЕЛ 2: ОБУЧЕНИЕ И БАЗОВАЯ ОЦЕНКА")
print("=" * 60)

# Для LogisticRegression нужно масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модели
models = {
    "RandomForest (без баланса)": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n{'-' * 40}")
    print(f"МОДЕЛЬ: {name}")
    print(f"{'-' * 40}")

    # Обучение
    if "Logistic" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc * 100:.2f}%)")

    # Полный classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    # Сохраняем результаты
    results[name] = {
        "accuracy": acc,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "model": model,
        "report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    }

    # Определяем худший класс по recall
    worst_recall = float('inf')
    worst_class = None
    worst_class_idx = None
    for idx, class_name in enumerate(target_names):
        recall = results[name]["report"][class_name]['recall']
        if recall < worst_recall:
            worst_recall = recall
            worst_class = class_name
            worst_class_idx = idx

    if worst_recall >= 1.0 - 1e-9:
        print(f"\nАнализ: Все классы предсказываются идеально (recall = 1.000)")
    else:
        print(f"\nАнализ: Хуже всего предсказывается класс '{worst_class}' "
              f"(recall = {worst_recall:.3f})")

        # Анализ матрицы ошибок - куда попадают ошибки
        cm_analysis = confusion_matrix(y_test, y_pred)
        errors_to_others = []
        for j in range(len(target_names)):
            if j != worst_class_idx:
                errors = cm_analysis[worst_class_idx, j]
                if errors > 0:
                    errors_to_others.append(f"    {errors} образцов → '{target_names[j]}'")

        if errors_to_others:
            print(f"  Ошибки класса '{worst_class}':")
            for err in errors_to_others:
                print(err)

# ─────────────────────────────────────────────────────────────────
# 3. СРАВНЕНИЕ И ВЫБОР СТРАТЕГИИ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("РАЗДЕЛ 3: СРАВНЕНИЕ И ВЫБОР СТРАТЕГИИ")
print("=" * 60)

# Сравнение macro F1
macro_f1_scores = {}
for name, res in results.items():
    macro_f1_scores[name] = res["report"]['macro avg']['f1-score']

print("\nСравнение macro F1-score:")
for name, f1 in macro_f1_scores.items():
    print(f"  {name}: {f1:.4f}")

best_model_name = max(macro_f1_scores, key=macro_f1_scores.get)
print(f"\nЛучшая модель по macro F1: {best_model_name} ({macro_f1_scores[best_model_name]:.4f})")

# Пробуем RandomForest с балансировкой
print("\n" + "-" * 40)
print("Пробуем RandomForest с class_weight='balanced'")
print("-" * 40)

rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

acc_bal = accuracy_score(y_test, y_pred_balanced)
report_bal = classification_report(y_test, y_pred_balanced, target_names=target_names, output_dict=True)
macro_f1_bal = report_bal['macro avg']['f1-score']

print(f"Accuracy: {acc_bal:.4f}")
print(f"Macro F1: {macro_f1_bal:.4f}")

# Анализ изменений для худшего класса - берем из обычного RandomForest
rf_report = results["RandomForest (без баланса)"]["report"]
worst_class_name = None
worst_recall_rf = float('inf')
for class_name in target_names:
    recall = rf_report[class_name]['recall']
    if recall < worst_recall_rf:
        worst_recall_rf = recall
        worst_class_name = class_name

# Проверяем, улучшился ли recall для этого класса
if worst_class_name is not None and worst_recall_rf < 1.0 - 1e-9:
    recall_worst_bal = report_bal[worst_class_name]['recall']
    print(f"\nВлияние балансировки на класс '{worst_class_name}':")
    print(f"  Recall до балансировки: {worst_recall_rf:.3f}")
    print(f"  Recall после балансировки: {recall_worst_bal:.3f}")
    print(f"  Изменение: {recall_worst_bal - worst_recall_rf:+.3f}")
else:
    print("\nВсе классы предсказываются идеально, балансировка не влияет на результат")
    recall_worst_bal = 1.0

# ─────────────────────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Распределение классов
axes[0, 0].bar(target_names, class_dist.values, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0, 0].set_title("Распределение классов в датасете")
axes[0, 0].set_ylabel("Количество образцов")
for i, v in enumerate(class_dist.values):
    axes[0, 0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# 2. Матрица ошибок RandomForest
ConfusionMatrixDisplay(
    results["RandomForest (без баланса)"]["confusion_matrix"],
    display_labels=target_names
).plot(ax=axes[0, 1], colorbar=False, cmap="Blues")
axes[0, 1].set_title("RandomForest (без баланса)")

# 3. Матрица ошибок LogisticRegression
ConfusionMatrixDisplay(
    results["LogisticRegression"]["confusion_matrix"],
    display_labels=target_names
).plot(ax=axes[0, 2], colorbar=False, cmap="Oranges")
axes[0, 2].set_title("LogisticRegression")

# 4. Матрица ошибок RandomForest balanced
cm_bal = confusion_matrix(y_test, y_pred_balanced)
ConfusionMatrixDisplay(cm_bal, display_labels=target_names).plot(
    ax=axes[1, 0], colorbar=False, cmap="Greens")
axes[1, 0].set_title("RandomForest (class_weight='balanced')")

# 5. Сравнение macro F1
models_for_compare = list(macro_f1_scores.keys()) + ["RF balanced"]
f1_values = list(macro_f1_scores.values()) + [macro_f1_bal]
colors_bar = ['#3498db', '#e74c3c', '#2ecc71']
bars = axes[1, 1].bar(models_for_compare, f1_values, color=colors_bar)
axes[1, 1].set_ylabel("Macro F1-score")
axes[1, 1].set_title("Сравнение моделей по macro F1")
axes[1, 1].set_ylim(0, 1)
for bar, val in zip(bars, f1_values):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=10)

# 6. Сравнение recall для худшего класса
if worst_class_name is not None:
    # Получаем recall для всех трех моделей по худшему классу
    recalls = [
        rf_report[worst_class_name]['recall'],
        results["LogisticRegression"]["report"][worst_class_name]['recall'],
        recall_worst_bal
    ]
    bars2 = axes[1, 2].bar(models_for_compare, recalls, color=colors_bar)
    axes[1, 2].set_ylabel(f"Recall для класса '{worst_class_name}'")
    axes[1, 2].set_title("Влияние на редкий класс")
    axes[1, 2].set_ylim(0, 1)
    for bar, val in zip(bars2, recalls):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', fontweight='bold')
else:
    axes[1, 2].text(0.5, 0.5, "Худший класс не определён",
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("Анализ recall")

plt.tight_layout()
plt.savefig("wine_classification_results.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────
# ИТОГОВЫЙ ВЫВОД
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)

print("\nЦель: выбрать модель для прототипа, если важно не пропустить представителей редкого класса.")
print("\nАнализ:")
print(f"• Без балансировки лучшая macro F1 у {best_model_name} ({macro_f1_scores[best_model_name]:.3f})")

# Проверяем, есть ли вообще проблемные классы
all_perfect = all(
    rf_report[cn]['recall'] >= 1.0 - 1e-9 for cn in target_names
) if 'rf_report' in dir() else macro_f1_scores[best_model_name] >= 1.0 - 1e-9

if all_perfect:
    print("• Все классы предсказываются идеально (recall = 1.000)")
    print("• Балансировка class_weight='balanced' не влияет на результат")
    print("\n" + "=" * 40)
    print("РЕКОМЕНДАЦИЯ:")
    print(f"  Для итогового прототипа выбираем {best_model_name}")
    print(f"   Причина: идеальный результат (macro F1 = {macro_f1_scores[best_model_name]:.3f})")
elif worst_class_name is not None:
    print(f"• Худший класс по recall: '{worst_class_name}' (recall = {worst_recall_rf:.3f} в RandomForest)")
    print(f"• RandomForest с balance: recall для '{worst_class_name}' вырос до {recall_worst_bal:.3f}")

    if recall_worst_bal > worst_recall_rf:
        print("\n  Балансировка улучшила распознавание редкого класса")

    print("\n" + "=" * 40)
    print("РЕКОМЕНДАЦИЯ:")
    if recall_worst_bal - worst_recall_rf > 0.05:
        print(f"  Для итогового прототипа выбираем RandomForest с class_weight='balanced'")
        print(f"   Причина: улучшение recall для редкого класса "
              f"(с {worst_recall_rf:.3f} до {recall_worst_bal:.3f})")
    else:
        print(f"  Для итогового прототипа выбираем {best_model_name}")
        print(f"   Причина: лучший баланс между классами (macro F1 = {macro_f1_scores[best_model_name]:.3f})")
else:
    print("• Обе модели показывают хорошие результаты")
    print("\n" + "=" * 40)
    print("РЕКОМЕНДАЦИЯ:")
    print(f"  Для итогового прототипа выбираем {best_model_name}")
    print(f"   Причина: лучшая macro F1 метрика ({macro_f1_scores[best_model_name]:.3f})")
print("=" * 40)