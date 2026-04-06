import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
)

# ─────────────────────────────────────────────────────────────────
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ (повтор из практики №1)
# ─────────────────────────────────────────────────────────────────
df = pd.read_csv("telecom_churn.csv")

# Автоопределение целевого столбца
churn_candidates = [c for c in df.columns if "churn" in c.lower()]
if not churn_candidates:
    raise ValueError("Не найден столбец с 'churn'. Укажите: target = 'ВашСтолбец'")
target = churn_candidates[0]
print(f"Целевой столбец: '{target}'")

X = df.drop(columns=[target])
y = df[target]
if y.dtype == bool or y.dtype == object:
    y = y.astype(int)

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ОБУЧЕНИЕ ТРЁХ БАЗОВЫХ МОДЕЛЕЙ (практика №1) — для сравнения
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
])

base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

base_results = {}
for name, model in base_models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    base_results[name] = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "pipeline":  pipe,
        "y_pred":    y_pred,
    }

best_base_name = max(base_results, key=lambda k: base_results[k]["accuracy"])
best_base_pipe = base_results[best_base_name]["pipeline"]
best_base_pred = base_results[best_base_name]["y_pred"]



# РАЗДЕЛ 1: ДИАГНОСТИКА ПРОБЛЕМЫ
print("\n" + "=" * 60)
print("РАЗДЕЛ 1: ДИАГНОСТИКА ПРОБЛЕМЫ")
print("=" * 60)

print(f"\nClassification Report ({best_base_name}):")
print(classification_report(y_test, best_base_pred,
                             target_names=["Остался (0)", "Ушёл (1)"]))

churn_rate = y.mean()
print(f"Дисбаланс классов: {(1-churn_rate)*100:.1f}% остались, {churn_rate*100:.1f}% ушли")
if churn_rate < 0.3:
    print("⚠ Дисбаланс есть — применим class_weight='balanced' и SMOTE")
else:
    print("Дисбаланс умеренный")

print("\nВывод: Для бизнеса критичнее False Negative (пропустить ушедшего клиента).")
print("Цель оптимизации: максимизировать Recall класса 'Ушёл' (F1-macro ≥ 0.75)")

fig_diag, axes_diag = plt.subplots(1, 2, figsize=(12, 4))

# Распределение классов
axes_diag[0].bar(["Остался (0)", "Ушёл (1)"],
                 y.value_counts().sort_index().values,
                 color=["#3498db", "#e74c3c"])
axes_diag[0].set_title("Распределение классов")
axes_diag[0].set_ylabel("Количество")

# Confusion matrix базовой модели
cm_base = confusion_matrix(y_test, best_base_pred)
ConfusionMatrixDisplay(cm_base, display_labels=["Остался", "Ушёл"]).plot(
    ax=axes_diag[1], colorbar=False, cmap="Blues")
axes_diag[1].set_title(f"Базовая матрица ошибок\n{best_base_name}")

plt.tight_layout()
plt.savefig("diag.png", dpi=150)
plt.show()

# ─────────────────────────────────────────────────────────────────
# РАЗДЕЛ 2: НАДЁЖНЫЙ КОНВЕЙЕР С БАЛАНСИРОВКОЙ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("РАЗДЕЛ 2: СОЗДАНИЕ НАДЁЖНОГО КОНВЕЙЕРА")
print("=" * 60)

# Пробуем SMOTE; если imblearn не установлен — используем class_weight
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote = SMOTE(random_state=42)

    # Предобрабатываем X_train для SMOTE (SMOTE работает только с числами)
    pre_fit = preprocessor.fit(X_train)
    X_train_pre = pre_fit.transform(X_train)
    X_test_pre  = pre_fit.transform(X_test)

    X_res, y_res = smote.fit_resample(X_train_pre, y_train)
    print(f"SMOTE применён. До: {len(y_train)} → После: {len(y_res)} строк")
    use_smote = True

except ImportError:
    print("imblearn не установлен → используем class_weight='balanced'")
    use_smote = False

# Выбираем оптимизируемую модель на основе победителя из практики №1
# Используем DecisionTree/LogReg с class_weight; KNN не поддерживает class_weight
if "KNeighbors" in best_base_name:
    opt_model_name = "Decision Tree"
    print(f"KNN не поддерживает class_weight → оптимизируем Decision Tree")
else:
    opt_model_name = best_base_name
    print(f"Оптимизируем: {opt_model_name}")

# ColumnTransformer (уже определён выше как preprocessor)
print("\nColumnTransformer:")
print(f"  Числовые признаки ({len(num_features)}): StandardScaler")
print(f"  Категориальные ({len(cat_features)}): OneHotEncoder")

# ─────────────────────────────────────────────────────────────────
# РАЗДЕЛ 3: ПОИСК ГИПЕРПАРАМЕТРОВ (GridSearchCV)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("РАЗДЕЛ 3: СИСТЕМНЫЙ ПОИСК ЛУЧШИХ ПАРАМЕТРОВ")
print("=" * 60)

if "Logistic" in opt_model_name:
    opt_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    param_grid = {
        "clf__C":       [0.01, 0.1, 1, 10],
        "clf__solver":  ["lbfgs", "liblinear"],
    }
else:  # Decision Tree
    opt_clf = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    param_grid = {
        "clf__max_depth":        [5, 10, 20, None],
        "clf__min_samples_leaf": [1, 5, 10],
        "clf__criterion":        ["gini", "entropy"],
    }

opt_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", opt_clf),
])

print(f"\nСетка параметров: {param_grid}")
print("Запуск GridSearchCV (5-fold, метрика: f1_macro)...")

grid_search = GridSearchCV(
    opt_pipe,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1,
)
grid_search.fit(X_train, y_train)

print(f"\nЛучшие параметры:  {grid_search.best_params_}")
print(f"Лучший F1 (CV):    {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────
# РАЗДЕЛ 4: ФИНАЛЬНАЯ ОЦЕНКА И ИНТЕРПРЕТАЦИЯ
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("РАЗДЕЛ 4: ФИНАЛЬНАЯ ОЦЕНКА И ИНТЕРПРЕТАЦИЯ")
print("=" * 60)

best_pipe = grid_search.best_estimator_
best_pipe.fit(X_train, y_train)  # обучаем на всех тренировочных данных
y_pred_opt = best_pipe.predict(X_test)

print("\nClassification Report (оптимизированная модель):")
print(classification_report(y_test, y_pred_opt,
                             target_names=["Остался (0)", "Ушёл (1)"]))

acc_opt  = accuracy_score(y_test, y_pred_opt)

# ─── ROC-кривая ───
y_prob = best_pipe.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC-ROC: {roc_auc:.4f}")

fig_final, axes_f = plt.subplots(1, 3, figsize=(18, 5))

# 1. ROC-кривая
axes_f[0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {roc_auc:.4f}")
axes_f[0].plot([0, 1], [0, 1], "k--", lw=1)
axes_f[0].set_xlabel("False Positive Rate")
axes_f[0].set_ylabel("True Positive Rate")
axes_f[0].set_title("ROC-кривая")
axes_f[0].legend(loc="lower right")

# 2. Confusion matrix оптимизированной модели
cm_opt = confusion_matrix(y_test, y_pred_opt)
ConfusionMatrixDisplay(cm_opt, display_labels=["Остался", "Ушёл"]).plot(
    ax=axes_f[1], colorbar=False, cmap="Oranges")
axes_f[1].set_title("Матрица ошибок (оптимизированная)")

# 3. Важность признаков (если есть)
clf_step = best_pipe.named_steps["clf"]
pre_step = best_pipe.named_steps["pre"]

if hasattr(clf_step, "feature_importances_"):
    # Восстанавливаем имена признаков после ColumnTransformer
    try:
        cat_enc = pre_step.named_transformers_["cat"]
        cat_names = cat_enc.get_feature_names_out(cat_features).tolist()
    except Exception:
        cat_names = [f"cat_{i}" for i in range(
            len(pre_step.named_transformers_["cat"].get_feature_names_out()))]
    feat_names = num_features + cat_names

    importances = clf_step.feature_importances_
    top_n = 15
    idx = np.argsort(importances)[::-1][:top_n]
    axes_f[2].barh(
        [feat_names[i] for i in reversed(idx)],
        importances[idx[::-1]],
        color="#2ecc71"
    )
    axes_f[2].set_title(f"Топ-{top_n} важных признаков")
    axes_f[2].set_xlabel("Feature Importance")
elif hasattr(clf_step, "coef_"):
    try:
        cat_enc = pre_step.named_transformers_["cat"]
        cat_names = cat_enc.get_feature_names_out(cat_features).tolist()
    except Exception:
        cat_names = []
    feat_names = num_features + cat_names
    coefs = np.abs(clf_step.coef_[0])
    top_n = 15
    idx = np.argsort(coefs)[::-1][:top_n]
    axes_f[2].barh(
        [feat_names[i] for i in reversed(idx)],
        coefs[idx[::-1]],
        color="#9b59b6"
    )
    axes_f[2].set_title(f"Топ-{top_n} коэффициентов (|coef|)")
    axes_f[2].set_xlabel("|Coefficient|")
else:
    axes_f[2].text(0.5, 0.5, "Важность признаков\nнедоступна для этой модели",
                   ha="center", va="center", transform=axes_f[2].transAxes)
    axes_f[2].set_title("Важность признаков")

plt.tight_layout()
plt.savefig("final_results.png", dpi=150)
plt.show()

# ─── Итоговая таблица сравнения ───
from sklearn.metrics import precision_score, recall_score, f1_score

def metrics_row(name, y_true, y_pred, y_prob_col=None):
    row = {
        "Модель":    name,
        "Accuracy":  f"{accuracy_score(y_true, y_pred):.4f}",
        "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
        "Recall":    f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
        "F1-macro":  f"{f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}",
    }
    if y_prob_col is not None:
        fpr_, tpr_, _ = roc_curve(y_true, y_prob_col)
        row["AUC-ROC"] = f"{auc(fpr_, tpr_):.4f}"
    else:
        row["AUC-ROC"] = "—"
    return row

base_prob = best_base_pipe.predict_proba(X_test)[:, 1] \
    if hasattr(best_base_pipe.named_steps["clf"], "predict_proba") else None

comparison = pd.DataFrame([
    metrics_row(f"Базовая ({best_base_name})", y_test, best_base_pred, base_prob),
    metrics_row(f"Оптимизированная ({opt_model_name})", y_test, y_pred_opt, y_prob),
])

print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
print("=" * 60)
print(comparison.to_string(index=False))
