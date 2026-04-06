import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# Проверка и установка imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote_available = True
except ImportError:
    print("⚠ imbalanced-learn не установлен. Установите: pip install imbalanced-learn")
    smote_available = False

# ─────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (как в практике 3)
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("ПРАКТИЧЕСКОЕ ЗАНЯТИЕ №4: ОПТИМИЗАЦИЯ МУЛЬТИКЛАССОВОЙ МОДЕЛИ")
print("=" * 60)

# Загрузка датасета
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
target_names = wine.target_names

print(f"\nДатасет Wine: {X.shape[0]} образцов, {X.shape[1]} признаков, {len(target_names)} классов")

# Разделение со стратификацией (сохраняем пропорции)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка:  {X_test.shape[0]} строк")

# ─────────────────────────────────────────────────────────────────
# БАЗОВАЯ МОДЕЛЬ (из практики 3)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 0: БАЗОВАЯ МОДЕЛЬ (RandomForest с class_weight='balanced')")
print("=" * 60)

base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

base_f1_macro = f1_score(y_test, y_pred_base, average='macro')
print(f"\nMacro F1 на тесте: {base_f1_macro:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_base, target_names=target_names))

# ─────────────────────────────────────────────────────────────────
# ШАГ 1: ПРИМЕНЕНИЕ SMOTE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 1: ПРИМЕНЕНИЕ SMOTE ДЛЯ БОРЬБЫ С ДИСБАЛАНСОМ")
print("=" * 60)

if smote_available:
    # SMOTE работает только с числовыми данными, масштабируем
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Применяем SMOTE только к тренировочным данным
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"\nДо SMOTE: {len(y_train)} образцов")
    print(f"После SMOTE: {len(y_train_smote)} образцов")

    # Распределение классов после SMOTE
    unique, counts = np.unique(y_train_smote, return_counts=True)
    print("Распределение классов после SMOTE:")
    for cls, count in zip(unique, counts):
        print(f"  Класс {cls} ({target_names[cls]}): {count} образцов")

    # Обучаем RandomForest на сбалансированных данных
    rf_smote = RandomForestClassifier(random_state=42)
    rf_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = rf_smote.predict(X_test_scaled)

    smote_f1_macro = f1_score(y_test, y_pred_smote, average='macro')
    print(f"\nMacro F1 на тесте (после SMOTE): {smote_f1_macro:.4f}")

    # Сравнение с базовой моделью
    improvement = (smote_f1_macro - base_f1_macro) * 100
    print(f"Изменение: {improvement:+.2f}%")

    best_y_pred = y_pred_smote
    best_model_name = "RandomForest + SMOTE"
    best_f1 = smote_f1_macro

    # Для дальнейшего использования
    X_train_best = X_train_smote
    y_train_best = y_train_smote
    use_scaled = True

else:
    print("\nSMOTE недоступен, используем class_weight='balanced' как базовую")
    best_y_pred = y_pred_base
    best_model_name = "RandomForest (class_weight='balanced')"
    best_f1 = base_f1_macro
    X_train_best = X_train
    y_train_best = y_train
    use_scaled = False

# ─────────────────────────────────────────────────────────────────
# ШАГ 2: ДИАГНОСТИКА С ПОМОЩЬЮ КРИВЫХ ОБУЧЕНИЯ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 2: ДИАГНОСТИКА МОДЕЛИ — КРИВЫЕ ОБУЧЕНИЯ")
print("=" * 60)

# Используем лучшую модель из предыдущего шага
if use_scaled:
    # Для SMOTE мы использовали масштабированные данные
    X_train_for_curve = X_train_scaled
    X_test_for_curve = X_test_scaled
    model_for_curve = RandomForestClassifier(random_state=42)
else:
    X_train_for_curve = X_train
    X_test_for_curve = X_test
    model_for_curve = RandomForestClassifier(random_state=42, class_weight='balanced')

# Строим кривые обучения
train_sizes, train_scores, val_scores = learning_curve(
    model_for_curve,
    X_train_for_curve, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
    random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

# Визуализация кривых обучения
fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1, color='blue')
ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                val_scores_mean + val_scores_std, alpha=0.1, color='orange')

ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Обучающая выборка (train)')
ax.plot(train_sizes, val_scores_mean, 'o-', color='orange', label='Кросс-валидация (validation)')

ax.set_xlabel('Размер обучающей выборки')
ax.set_ylabel('Macro F1-score')
ax.set_title('Кривые обучения RandomForest')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()

# Анализ кривых обучения
final_gap = train_scores_mean[-1] - val_scores_mean[-1]
print(f"\nАнализ кривых обучения:")
print(f"  Финальная разница train-val: {final_gap:.4f}")

if final_gap > 0.1:
    print("  ⚠ Признаки переобучения: train > val более чем на 0.1")
    print("     → Рекомендация: упростить модель (уменьшить max_depth, увеличить min_samples_leaf)")
elif train_scores_mean[-1] < 0.8 and val_scores_mean[-1] < 0.8:
    print("  ⚠ Модель недообучена: оба показателя ниже 0.8")
    print("     → Рекомендация: увеличить сложность модели (больше n_estimators, меньше ограничений)")
elif train_scores_mean[-1] - val_scores_mean[-1] < 0.05:
    print("  ✓ Модель хорошо сбалансирована: gap < 0.05")
    print("     → Можно попробовать собрать больше данных для улучшения")
else:
    print("  → Кривые обучения в норме")

# ─────────────────────────────────────────────────────────────────
# ШАГ 3: GRIDSEARCHCV — КОМПЛЕКСНАЯ НАСТРОЙКА
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ШАГ 3: GRIDSEARCHCV — КОМПЛЕКСНАЯ НАСТРОЙКА ГИПЕРПАРАМЕТРОВ")
print("=" * 60)

# Создаем Pipeline
if smote_available:
    # Используем ImbPipeline, чтобы SMOTE применялся только к тренировочным фолдам
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42))
    ])
else:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
X_train_grid = X_train

# Сетка гиперпараметров
param_grid = {
    'rf__n_estimators': [50, 100],
    'rf__max_depth': [5, 10, None],
    'rf__min_samples_leaf': [1, 2, 4]
}

print("\nСетка гиперпараметров:")
print(f"  n_estimators: {param_grid['rf__n_estimators']}")
print(f"  max_depth: {param_grid['rf__max_depth']}")
print(f"  min_samples_leaf: {param_grid['rf__min_samples_leaf']}")
print(
    f"  Всего комбинаций: {len(param_grid['rf__n_estimators']) * len(param_grid['rf__max_depth']) * len(param_grid['rf__min_samples_leaf'])}")

# StratifiedKFold для стратифицированной кросс-валидации
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\nЗапуск GridSearchCV (3-fold, scoring=f1_macro)...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_grid, y_train)

print(f"\nЛучшие параметры: {grid_search.best_params_}")
print(f"Лучший macro F1 (CV): {grid_search.best_score_:.4f}")

# Оценка на тестовых данных
best_model = grid_search.best_estimator_
y_pred_final = best_model.predict(X_test)

final_f1_macro = f1_score(y_test, y_pred_final, average='macro')
final_accuracy = accuracy_score(y_test, y_pred_final)

print(f"\nФинальная оценка на тесте:")
print(f"  Macro F1: {final_f1_macro:.4f}")
print(f"  Accuracy: {final_accuracy:.4f}")

# ─────────────────────────────────────────────────────────────────
# ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ")
print("=" * 60)

# Собираем результаты по шагам
if smote_available:
    results_table = pd.DataFrame([
        {"Шаг": "Базовая модель (RF + class_weight)", "Macro F1 (test)": f"{base_f1_macro:.4f}"},
        {"Шаг": "+ SMOTE (без GridSearch)", "Macro F1 (test)": f"{smote_f1_macro:.4f}"},
        {"Шаг": "+ SMOTE + GridSearchCV", "Macro F1 (test)": f"{final_f1_macro:.4f}"}
    ])
else:
    results_table = pd.DataFrame([
        {"Шаг": "Базовая модель (RF + class_weight)", "Macro F1 (test)": f"{base_f1_macro:.4f}"},
        {"Шаг": "+ GridSearchCV", "Macro F1 (test)": f"{final_f1_macro:.4f}"}
    ])

print(results_table.to_string(index=False))

# Общее улучшение
total_improvement = (final_f1_macro - base_f1_macro) * 100
print(f"\nОбщее улучшение: {total_improvement:+.2f}%")

if base_f1_macro >= 0.999:
    print("✓ Базовая модель уже достигла идеального результата (F1 = 1.0)")
    print("  Датасет Wine хорошо разделим — дальнейшая оптимизация не требуется")
elif total_improvement >= 5:
    print("✓ Цель достигнута: улучшение макро-F1 на 5% и более")
elif total_improvement > 0:
    print("✓ Улучшение достигнуто, но менее 5%")
else:
    print("⚠ Улучшение не достигнуто, возможно, требуется другой подход")

# ─────────────────────────────────────────────────────────────────
# ФИНАЛЬНАЯ ВИЗУАЛИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Матрица ошибок финальной модели
cm_final = confusion_matrix(y_test, y_pred_final)
ConfusionMatrixDisplay(cm_final, display_labels=target_names).plot(
    ax=axes[0], colorbar=False, cmap="Greens")
axes[0].set_title(f"Матрица ошибок\nФинальная модель (Macro F1 = {final_f1_macro:.3f})")

# 2. Сравнение Macro F1 по шагам
steps = ["Базовая", "SMOTE", "GridSearch"]
if smote_available:
    f1_values = [base_f1_macro, smote_f1_macro, final_f1_macro]
else:
    steps = ["Базовая", "GridSearch"]
    f1_values = [base_f1_macro, final_f1_macro]

bars = axes[1].bar(steps, f1_values, color=['#3498db', '#2ecc71', '#e74c3c'][:len(steps)])
axes[1].set_ylabel("Macro F1-score")
axes[1].set_title("Улучшение модели по шагам")
axes[1].set_ylim(0, 1)
for bar, val in zip(bars, f1_values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontweight='bold')

# 3. Важность признаков финальной модели
best_rf = best_model.named_steps['rf']
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

axes[2].barh(range(10), importances[indices][::-1], color='#9b59b6')
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([wine.feature_names[i] for i in indices[::-1]])
axes[2].set_xlabel("Важность признака")
axes[2].set_title("Топ-10 наиболее важных признаков")

plt.tight_layout()
plt.savefig("final_comparison.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────
# ФИНАЛЬНЫЙ ВЫВОД
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ВЫВОДЫ ПО РАБОТЕ")
print("=" * 60)

print("""
1. SMOTE:
   - Синтезирует новые образцы миноритарных классов
   - Позволяет модели лучше обучаться на редких классах

2. Кривые обучения:
   - Показывают разрыв между train и validation
   - Помогают диагностировать переобучение/недообучение

3. GridSearchCV:
   - Автоматически подбирает лучшие гиперпараметры
   - Использует стратифицированную кросс-валидацию

4. Итог:""")

if base_f1_macro >= 0.999:
    print(f"   ✓ Базовая модель уже показывает F1 = {base_f1_macro:.2f}")
    print("     Датасет Wine хорошо разделим для RandomForest, потолок достигнут")
elif total_improvement >= 5:
    print(f"   ✓ Цель достигнута! Macro F1 улучшен на {total_improvement:.1f}%")
elif total_improvement > 0:
    print(f"   ✓ Модель улучшена на {total_improvement:.1f}%")
else:
    print("   → Для дальнейшего улучшения требуется другой подход или больше данных")

print("\n" + "=" * 60)