import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# ─────────────────────────────────────────────
# 1. РАЗБОР ДАННЫХ
# ─────────────────────────────────────────────
df = pd.read_csv("telecom_churn.csv")

print("=" * 50)
print("РАЗДЕЛ 1: РАЗБОР ДАННЫХ")
print("=" * 50)

print(f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")

# ── Диагностика: все столбцы ──
print("\nВсе столбцы датасета:")
print(df.columns.tolist())

# ── Автоопределение целевого столбца (регистронезависимо) ──
churn_candidates = [c for c in df.columns if "churn" in c.lower()]
if not churn_candidates:
    raise ValueError(
        "Не найден столбец с 'churn' в названии.\n"
        "Укажите имя вручную: target = 'ВашСтолбец'"
    )
target = churn_candidates[0]
print(f"\nЦелевой столбец определён автоматически: '{target}'")

print("\nЧисловые столбцы:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(numeric_cols)

print("\nКатегориальные столбцы:")
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
print(categorical_cols)

print("\nПропуски по столбцам:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "Пропусков нет")

print(f"\nРаспределение целевой переменной '{target}':")
print(df[target].value_counts())
print(df[target].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

# ─────────────────────────────────────────────
# 2. ПОДГОТОВКА ДАННЫХ
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("РАЗДЕЛ 2: ПОДГОТОВКА ДАННЫХ")
print("=" * 50)

X = df.drop(columns=[target])
y = df[target]

# Приводим к int если булев или строковый тип
if y.dtype == bool or y.dtype == object:
    y = y.astype(int)

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print(f"\nЧисловых признаков: {len(num_features)}")
print(f"Категориальных признаков: {len(cat_features)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nОбучающая выборка: {X_train.shape[0]} строк")
print(f"Тестовая выборка:  {X_test.shape[0]} строк")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
])

# ─────────────────────────────────────────────
# 3. ОБУЧЕНИЕ И ПРОВЕРКА МОДЕЛЕЙ
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("РАЗДЕЛ 3: ОБУЧЕНИЕ И ПРОВЕРКА МОДЕЛЕЙ")
print("=" * 50)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
}

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"accuracy": acc, "pipeline": pipeline, "y_pred": y_pred}
    print(f"\n{name}:")
    print(f"  Accuracy = {acc:.4f} ({acc*100:.2f}%)")

# ─────────────────────────────────────────────
# 4. АНАЛИЗ РЕЗУЛЬТАТОВ
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("РАЗДЕЛ 4: АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 50)

acc_df = pd.DataFrame([
    {"Модель": name, "Accuracy": f"{v['accuracy']:.4f}", "Accuracy (%)": f"{v['accuracy']*100:.2f}%"}
    for name, v in results.items()
]).sort_values("Accuracy", ascending=False).reset_index(drop=True)

print("\nСравнение моделей:")
print(acc_df.to_string(index=False))

best_name = max(results, key=lambda k: results[k]["accuracy"])
print(f"\nЛучшая модель: {best_name} (Accuracy = {results[best_name]['accuracy']*100:.2f}%)")

# ─── График сравнения + матрица ошибок ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names  = list(results.keys())
accs   = [results[n]["accuracy"] for n in names]
colors = ["#2ecc71" if n == best_name else "#3498db" for n in names]

axes[0].bar(names, accs, color=colors, edgecolor="white", linewidth=1.2)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Сравнение точности моделей")
for i, (name, acc) in enumerate(zip(names, accs)):
    axes[0].text(i, acc + 0.01, f"{acc:.4f}", ha="center", va="bottom", fontweight="bold")
axes[0].tick_params(axis="x", rotation=10)

best_pred = results[best_name]["y_pred"]
cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Остался", "Ушёл"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title(f"Матрица ошибок — {best_name}")

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nМатрица ошибок ({best_name}):")
print(f"  Верно предсказано «Остался» (TN): {tn}")
print(f"  Верно предсказано «Ушёл»    (TP): {tp}")
print(f"  Ошибочно предсказано «Ушёл» (FP): {fp}  ← лояльных клиентов модель приняла за ушедших")
print(f"  Ошибочно предсказано «Остался» (FN): {fn}  ← ушедших клиентов модель не распознала")