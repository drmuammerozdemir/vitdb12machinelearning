# app.py
# Streamlit ML app: Hemogram -> predict B12 and Vitamin D (regression)
# Author: you
# Run: streamlit run app.py

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Baseline / linear
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor

# Trees / ensembles
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

# Optional: XGBoost / LightGBM / CatBoost (will be enabled if installed)
XGBRegressor = None
LGBMRegressor = None
CatBoostRegressor = None

try:
    from xgboost import XGBRegressor as _XGBRegressor
    XGBRegressor = _XGBRegressor
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor as _LGBMRegressor
    LGBMRegressor = _LGBMRegressor
except Exception:
    pass

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor
    CatBoostRegressor = _CatBoostRegressor
except Exception:
    pass

st.set_page_config(page_title="Hemogram → B12 & Vit D Tahmini", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def normalize_colname(c: str) -> str:
    """Standardize columns: trim, fix special chars/spaces, upper."""
    c2 = c.strip()
    c2 = re.sub(r"\s+", " ", c2)
    return c2

def safe_to_numeric(s: pd.Series) -> pd.Series:
    """Convert with Turkish decimal commas etc."""
    # handle strings like "12,3" -> "12.3"
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")

def compute_metrics(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
    }

def get_feature_groups(columns: list[str]) -> dict:
    """Define your feature groups. Hemogram: WBC’ye kadar (WBC dahil)."""
    # Based on your provided schema:
    # PROTOKOL_NO, BA#, BA%, EO#, EO%, HCT, HGB, LY#, LY%, MCH, MCHC, MCV, MO#, MO%, MPV,
    # NE#, NE%, PCT, PDW, PLT, RBC, RDW-CV, RDW-SD, WBC, B12, VİTAMİN D, CINSIYET, HASTA_YAS, NLR, PLR, LMR

    hemogram_until_wbc = [
        "BA#", "BA%", "EO#", "EO%", "HCT", "HGB", "LY#", "LY%", "MCH", "MCHC", "MCV",
        "MO#", "MO%", "MPV", "NE#", "NE%", "PCT", "PDW", "PLT", "RBC", "RDW-CV", "RDW-SD", "WBC"
    ]
    derived = ["NLR", "PLR", "LMR"]
    demo = ["CINSIYET", "HASTA_YAS"]

    present = set(columns)
    return {
        "Sadece hemogram (WBC dahil)": [c for c in hemogram_until_wbc if c in present],
        "Hemogram + demografi": [c for c in hemogram_until_wbc if c in present] + [c for c in demo if c in present],
        "Hemogram + demografi + türev oranlar": [c for c in hemogram_until_wbc if c in present] + [c for c in demo if c in present] + [c for c in derived if c in present],
        "Tüm uygun özellikler (ID hariç)": [c for c in columns if c not in {"PROTOKOL_NO", "B12", "VİTAMİN D"}],
    }

def build_model(model_name: str, seed: int):
    """Return an sklearn-compatible regressor instance."""
    if model_name == "LinearRegression":
        return LinearRegression()
    if model_name == "Ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if model_name == "Lasso":
        return Lasso(alpha=0.001, random_state=seed, max_iter=5000)
    if model_name == "ElasticNet":
        return ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=seed, max_iter=5000)
    if model_name == "HuberRegressor (robust)":
        return HuberRegressor()

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=500, random_state=seed, n_jobs=-1, max_depth=None, min_samples_leaf=2
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=800, random_state=seed, n_jobs=-1, max_depth=None, min_samples_leaf=2
        )
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=seed)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=seed)

    if model_name == "XGBoost (if installed)":
        if XGBRegressor is None:
            raise RuntimeError("xgboost yüklü değil. requirements.txt'e ekleyin.")
        return XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            objective="reg:squarederror",
        )

    if model_name == "LightGBM (if installed)":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm yüklü değil. requirements.txt'e ekleyin.")
        return LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "CatBoost (if installed)":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost yüklü değil. requirements.txt'e ekleyin.")
        return CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            random_seed=seed,
            loss_function="RMSE",
            verbose=False
        )

    raise ValueError("Bilinmeyen model seçimi.")

def build_pipeline(X: pd.DataFrame, model, scale_numeric: bool = False) -> Pipeline:
    """Preprocess numeric/categorical + model."""
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_cols),
            ("cat", Pipeline(categorical_steps), categorical_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]

    # Convert known numeric columns to numeric
    for col in df.columns:
        if col in {"CINSIYET"}:
            continue
        if col in {"PROTOKOL_NO"}:
            # keep as string (ID) to avoid leakage; users can drop it later anyway
            df[col] = df[col].astype(str)
            continue
        # attempt numeric conversion for all others
        df[col] = safe_to_numeric(df[col]) if (df[col].dtype == object or "VİTAMİN" in col or col in {"B12", "HASTA_YAS"}) else df[col]

    # Standardize gender if present
    if "CINSIYET" in df.columns:
        df["CINSIYET"] = df["CINSIYET"].astype(str).str.strip().str.upper()
        df["CINSIYET"] = df["CINSIYET"].replace({
            "ERKEK": "E", "KADIN": "K", "MALE": "E", "FEMALE": "K"
        })

    return df


# -----------------------------
# UI
# -----------------------------
st.title("Hemogram ile B12 ve Vitamin D Tahmini (Regresyon)")

with st.sidebar:
    st.header("Veri")
    uploaded = st.file_uploader("CSV yükle", type=["csv"])
    sep = st.selectbox("CSV ayırıcı", [",", ";", "\t"], index=0)
    encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "cp1254", "latin1"], index=1)

    st.divider()
    st.header("Model")
    seed = st.number_input("Random seed", value=42, step=1)
    test_size = st.slider("Test oranı", 0.1, 0.4, 0.2, 0.05)

    available_models = [
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "HuberRegressor (robust)",
        "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
        "XGBoost (if installed)", "LightGBM (if installed)", "CatBoost (if installed)"
    ]
    model_name = st.selectbox("Model seç", available_models, index=6)

    scale_numeric = st.checkbox("Sayısal değişkenleri ölçekle (Linear modeller için iyi)", value=False)

    st.divider()
    st.header("Hedef & Özellikler")
    target_choice = st.radio("Hedef", ["B12", "VİTAMİN D"], index=0)
    do_multitarget = st.checkbox("İkisini aynı anda değerlendir (B12 + Vit D raporu)", value=True)

    st.divider()
    st.header("Değerlendirme")
    cv_folds = st.slider("CV fold", 3, 10, 5, 1)
    do_perm_importance = st.checkbox("Permutation importance hesapla (daha yavaş)", value=True)
    perm_repeats = st.slider("Permutation tekrar", 3, 20, 8, 1)

st.caption("Not: Bu uygulama klinik karar aracı değildir; araştırma/hipotez amaçlıdır.")

if uploaded is None:
    st.info("Başlamak için CSV dosyanı yükle. Kolonlar: hemogram (WBC’ye kadar), B12, VİTAMİN D, CINSIYET, HASTA_YAS, NLR/PLR/LMR vb.")
    st.stop()

# Read
try:
    import csv
from io import StringIO

def robust_read_csv(uploaded_file, encoding: str):
    # Dosyayı text olarak al
    raw_bytes = uploaded_file.getvalue()
    text = raw_bytes.decode(encoding, errors="replace")

    # 1) Ayracı otomatik tahmin etmeye çalış (sniffer)
    sample = text[:20000]
    guessed_sep = None
    try:
        guessed_sep = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        pass

    # 2) Denenecek ayraçlar listesi
    seps_to_try = []
    if guessed_sep:
        seps_to_try.append(guessed_sep)
    seps_to_try += [";", ",", "\t", "|"]
    # unique sırayı koru
    seen = set()
    seps_to_try = [s for s in seps_to_try if not (s in seen or seen.add(s))]

    best_df = None
    best_score = -1
    best_sep = None
    best_bad_lines = None

    for s in seps_to_try:
        bad_lines = []
        try:
            # on_bad_lines callable: bozuk satırları yakala
            def bad_line_handler(line):
                bad_lines.append(line)
                return None  # satırı at

            df = pd.read_csv(
                StringIO(text),
                sep=s,
                engine="python",          # toleranslı parser
                on_bad_lines=bad_line_handler,
                quoting=csv.QUOTE_MINIMAL
            )

            # skor: dol

except Exception as e:
    st.error(f"CSV okunamadı: {e}")
    st.stop()

df = clean_dataframe(df_raw)

st.subheader("Veri Önizleme")
st.write(df.head(10))

required_targets = {"B12", "VİTAMİN D"}
missing_targets = required_targets - set(df.columns)
if missing_targets:
    st.error(f"Hedef sütun(lar) eksik: {missing_targets}. CSV kolon isimlerini kontrol et.")
    st.stop()

# Feature group selection
groups = get_feature_groups(df.columns.tolist())
group_name = st.selectbox("Özellik seti", list(groups.keys()), index=0)
feature_cols = groups[group_name]

if not feature_cols:
    st.error("Seçilen özellik setinde hiç sütun bulunamadı. Kolon isimlerini kontrol et.")
    st.stop()

# Filter: drop rows with missing target
df_model = df.copy()
df_model = df_model.dropna(subset=[target_choice])

# Basic sanity filters (optional)
# Example: keep age <= 16 if exists
if "HASTA_YAS" in df_model.columns:
    df_model = df_model[df_model["HASTA_YAS"].between(0, 16, inclusive="both")]

st.write(f"Modelleme için örnek sayısı: **{len(df_model):,}**")

# Prepare X, y
X = df_model[feature_cols].copy()

# ensure numeric columns are numeric
for c in X.columns:
    if c != "CINSIYET" and X[c].dtype == object:
        X[c] = safe_to_numeric(X[c])

y_b12 = df_model["B12"].copy()
y_vd  = df_model["VİTAMİN D"].copy()

# Train/test split (single target)
X_train, X_test, y_train, y_test = train_test_split(
    X, df_model[target_choice], test_size=float(test_size), random_state=int(seed)
)

# Build model + pipeline
try:
    reg = build_model(model_name, int(seed))
except Exception as e:
    st.error(str(e))
    st.stop()

pipe = build_pipeline(X_train, reg, scale_numeric=scale_numeric)

# Cross-validation
st.subheader("Çapraz Doğrulama (CV)")
cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}

with st.spinner("CV hesaplanıyor..."):
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

cv_r2 = np.mean(cv_res["test_r2"])
cv_mae = -np.mean(cv_res["test_mae"])
cv_rmse = -np.mean(cv_res["test_rmse"])

c1, c2, c3 = st.columns(3)
c1.metric("CV R²", f"{cv_r2:.3f}")
c2.metric("CV MAE", f"{cv_mae:.3f}")
c3.metric("CV RMSE", f"{cv_rmse:.3f}")

# Fit + test
st.subheader("Test Sonuçları")
with st.spinner("Model eğitiliyor..."):
    pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
m = compute_metrics(y_test, y_pred)

c1, c2, c3 = st.columns(3)
c1.metric("Test R²", f"{m['R2']:.3f}")
c2.metric("Test MAE", f"{m['MAE']:.3f}")
c3.metric("Test RMSE", f"{m['RMSE']:.3f}")

# Residual view
res_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
res_df["residual"] = res_df["y_true"] - res_df["y_pred"]
st.write(res_df.head(20))

# Permutation importance
if do_perm_importance:
    st.subheader("Özellik Önemi (Permutation Importance)")
    with st.spinner("Permutation importance hesaplanıyor (biraz sürebilir)..."):
        # Note: permutation_importance needs raw X_test, y_test
        r = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=int(perm_repeats),
            random_state=int(seed),
            n_jobs=-1,
            scoring="r2",
        )
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)

    st.dataframe(imp, use_container_width=True)

# Multi-target evaluation (separate fits; simple & clear)
if do_multitarget:
    st.subheader("B12 + Vitamin D (İki hedef ayrı ayrı rapor)")
    targets = ["B12", "VİTAMİN D"]
    report_rows = []
    for t in targets:
        df_tmp = df.copy().dropna(subset=[t])
        if "HASTA_YAS" in df_tmp.columns:
            df_tmp = df_tmp[df_tmp["HASTA_YAS"].between(0, 16, inclusive="both")]
        X2 = df_tmp[feature_cols].copy()
        for c in X2.columns:
            if c != "CINSIYET" and X2[c].dtype == object:
                X2[c] = safe_to_numeric(X2[c])
        y2 = df_tmp[t]
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            X2, y2, test_size=float(test_size), random_state=int(seed)
        )
        pipe2 = build_pipeline(X2_train, build_model(model_name, int(seed)), scale_numeric=scale_numeric)
        pipe2.fit(X2_train, y2_train)
        pred2 = pipe2.predict(X2_test)
        met = compute_metrics(y2_test, pred2)
        report_rows.append({
            "target": t,
            "n": len(df_tmp),
            "R2": met["R2"],
            "MAE": met["MAE"],
            "RMSE": met["RMSE"],
        })
    report = pd.DataFrame(report_rows)
    st.dataframe(report, use_container_width=True)

st.success("Bitti. İstersen bir sonraki adımda hiperparametre optimizasyonu (Optuna) ve SHAP açıklanabilirlik ekleyelim.")
