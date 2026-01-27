# app.py
# Streamlit ML app: Hemogram -> predict B12 and Vitamin D (regression)
# Run: streamlit run app.py

from __future__ import annotations

import re
import os
import csv
from io import StringIO, BytesIO
from scipy.stats import kruskal, f_oneway, shapiro, ttest_ind, mannwhitneyu

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import streamlit as st


from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Scikit-learn 1.6+ uyumlu metrikler (RMSE manuel hesaplanacak)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# Baseline / linear
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor

# Trees / ensembles
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

# Optional libraries check
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

st.set_page_config(page_title="Hemogram -> B12 & Vit D Tahmini", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def normalize_colname(c: str) -> str:
    c2 = c.strip()
    c2 = re.sub(r"\s+", " ", c2)
    return c2


def safe_to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False)
        s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    return pd.to_numeric(s, errors="coerce")


def compute_metrics(y_true, y_pred) -> dict:
    # KRİTİK DÜZELTME: squared=False parametresi KALDIRILDI.
    # RMSE'yi manuel hesaplıyoruz. Bu yöntem her versiyonda çalışır.
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"R2": r2_score(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred), "RMSE": rmse}


# RAM KORUMASI: Bu fonksiyonu cache'e alıyoruz ki her defasında hesaplayıp sistemi yormasın
@st.cache_data(show_spinner=False)
def calculate_permutation_importance(_pipe, X_val, y_val, repeats, seed):
    # n_jobs=1 yaparak RAM patlamasını önlüyoruz (Streamlit Cloud için şart)
    r = permutation_importance(
        _pipe, X_val, y_val,
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,  # <--- BURASI ÇOK ÖNEMLİ (Eskiden -1 idi)
        scoring="r2"
    )
    return r


def get_feature_groups(columns: list[str]) -> dict:
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
        "Hemogram + demografi + türev oranlar": (
            [c for c in hemogram_until_wbc if c in present]
            + [c for c in demo if c in present]
            + [c for c in derived if c in present]
        ),
        "Tüm uygun özellikler (ID hariç)": [c for c in columns if c not in {"PROTOKOL_NO", "B12", "VİTAMİN D"}],
    }


def build_model(model_name: str, seed: int):
    # Ağaç tabanlı modellerde de n_jobs=1 yaparak fit sırasında çökme riskini azaltıyoruz
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
            n_estimators=300, random_state=seed, n_jobs=1, max_depth=None, min_samples_leaf=2
        )
    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=500, random_state=seed, n_jobs=1, max_depth=None, min_samples_leaf=2
        )
    if model_name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=seed)
    if model_name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=seed)

    if model_name == "XGBoost (if installed)":
        if XGBRegressor is None:
            raise RuntimeError("xgboost yüklü değil. requirements.txt'e ekleyin.")
        return XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1, # RAM Koruması
            objective="reg:squarederror",
        )

    if model_name == "LightGBM (if installed)":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm yüklü değil. requirements.txt'e ekleyin.")
        return LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1, # RAM Koruması
        )

    if model_name == "CatBoost (if installed)":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost yüklü değil. requirements.txt'e ekleyin.")
        return CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            random_seed=seed,
            loss_function="RMSE",
            verbose=False,
            thread_count=1 # RAM Koruması
        )

    raise ValueError("Bilinmeyen model seçimi.")


def build_pipeline(X: pd.DataFrame, model, scale_numeric: bool = False) -> Pipeline:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
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

    return Pipeline([("pre", pre), ("model", model)])


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Önce sütun isimlerini temizle
    df.columns = [normalize_colname(c) for c in df.columns]

    # Sayısal Dönüşümler
    for col in df.columns:
        if col == "CINSIYET":
            continue
        if col == "PROTOKOL_NO":
            df[col] = df[col].astype(str)
            continue
        # B12, Vitamin D ve Yaş dahil sayısal yap
        if df[col].dtype == object or "VİTAMİN" in col or col in {"B12", "HASTA_YAS"}:
            df[col] = safe_to_numeric(df[col])

    # CİNSİYET DÜZELTMESİ (1 -> E, 2 -> K)
    if "CINSIYET" in df.columns:
        # Önce string'e çevir, varsa .0'ları at (Excel bazen 1.0 diye okur)
        s = df["CINSIYET"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        
        # Haritalama yap
        mapping = {"1": "E", "2": "K", "ERKEK": "E", "KADIN": "K", "MALE": "E", "FEMALE": "K"}
        df["CINSIYET"] = s.map(mapping).fillna(s) # Eşleşmezse eski halini koru
        
        # Son temizlik
        df["CINSIYET"] = df["CINSIYET"].astype(str).str.upper()

    return df

def calculate_derived_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hemogram parametrelerinden türetilmiş indeksleri hesaplar.
    (Sıfıra bölünme hatalarını np.nan ile engeller)
    """
    df = df.copy()
    
    # Gerekli sütunların varlığını kontrol et (Normalize edilmiş isimlerle)
    # Genelde: NE#, LY#, MO#, PLT, RBC, MCV
    
    # Yardımcı lambda: Güvenli bölme
    safe_div = lambda a, b: a / b if b != 0 else np.nan

    # Vektörel işlem için numpy kullanımı daha hızlıdır
    ne = df.get("NE#", np.nan)
    ly = df.get("LY#", np.nan)
    mo = df.get("MO#", np.nan)
    plt = df.get("PLT", np.nan)
    rbc = df.get("RBC", np.nan)
    mcv = df.get("MCV", np.nan)
    rdw = df.get("RDW-CV", np.nan) # Veya RDW-SD

    # 1. NLR (Neutrophil-to-Lymphocyte Ratio)
    if "NLR" not in df.columns and "NE#" in df.columns and "LY#" in df.columns:
        df["NLR"] = ne / ly

    # 2. PLR (Platelet-to-Lymphocyte Ratio)
    if "PLR" not in df.columns and "PLT" in df.columns and "LY#" in df.columns:
        df["PLR"] = plt / ly

    # 3. LMR (Lymphocyte-to-Monocyte Ratio)
    if "LMR" not in df.columns and "LY#" in df.columns and "MO#" in df.columns:
        df["LMR"] = ly / mo

    # 4. SII (Systemic Immune-Inflammation Index) = (PLT x NE) / LY
    if "SII" not in df.columns and "PLT" in df.columns and "NE#" in df.columns and "LY#" in df.columns:
        df["SII"] = (plt * ne) / ly

    # 5. SIRI (Systemic Inflammation Response Index) = (NE x MO) / LY
    if "SIRI" not in df.columns and "NE#" in df.columns and "MO#" in df.columns and "LY#" in df.columns:
        df["SIRI"] = (ne * mo) / ly
        
    # 6. AISI (Aggregate Index of Systemic Inflammation) = (NE x PLT x MO) / LY
    if "AISI" not in df.columns and "NE#" in df.columns and "PLT" in df.columns and "MO#" in df.columns:
        df["AISI"] = (ne * plt * mo) / ly

    # 7. Mentzer Index (Talasemi Taraması) = MCV / RBC (<13 Talasemi, >13 Demir Eksikliği)
    if "Mentzer" not in df.columns and "MCV" in df.columns and "RBC" in df.columns:
        df["Mentzer"] = mcv / rbc

    # Sonsuz değerleri (inf) NaN yapalım
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def segment_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    HASTA_YAS sütununa göre pediyatrik gruplama yapar.
    0-5: Okul Öncesi
    6-11: Okul Çağı
    12-17: Adolesan
    """
    if "HASTA_YAS" not in df.columns:
        return df
    
    # cut fonksiyonunda bins aralıkları: (dahil değil, dahil] mantığıyla çalışır ama include_lowest=True ile ilkini de alırız.
    # Ancak manuel mantık daha hatasız çalışır burada.
    
    conditions = [
        (df['HASTA_YAS'] >= 0) & (df['HASTA_YAS'] <= 5),
        (df['HASTA_YAS'] >= 6) & (df['HASTA_YAS'] <= 11),
        (df['HASTA_YAS'] >= 12) & (df['HASTA_YAS'] <= 17)
    ]
    choices = ['Okul Öncesi (0-5)', 'Okul Çağı (6-11)', 'Adolesan (12-17)']
    
    df['Yas_Grubu'] = np.select(conditions, choices, default='Diğer')
    return df

#-------------B12 VE D VİTAMİNİ SEVİYELERİNE GÖRE SINIFLAMA YAPMA-------------#
def segment_clinical_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    B12 ve Vitamin D seviyelerine göre gruplama yapar.
    DÜZELTME: np.select içinde default değer 'np.nan' yerine 'Diğer' yapıldı (TypeError hatası için).
    """
    # --- B12 GRUPLAMA (<200, 200-400, >400) ---
    if "B12" in df.columns:
        conditions_b12 = [
            (df['B12'] < 200),
            (df['B12'] >= 200) & (df['B12'] <= 400),
            (df['B12'] > 400)
        ]
        choices_b12 = ['1. Düşük (<200)', '2. Sınırda (200-400)', '3. Yüksek (>400)']
        
        # HATA DÜZELTİLDİ: default=np.nan yerine default='Diğer'
        df['B12_Grubu'] = np.select(conditions_b12, choices_b12, default='Diğer')

    # --- VITAMIN D GRUPLAMA (<20, 20-30, >30) ---
    if "VİTAMİN D" in df.columns:
        conditions_vitd = [
            (df['VİTAMİN D'] < 20),
            (df['VİTAMİN D'] >= 20) & (df['VİTAMİN D'] <= 30),
            (df['VİTAMİN D'] > 30)
        ]
        choices_vitd = ['1. Eksiklik (<20)', '2. Yetersizlik (20-30)', '3. Yeterli (>30)']
        
        # HATA DÜZELTİLDİ: default=np.nan yerine default='Diğer'
        df['VitD_Grubu'] = np.select(conditions_vitd, choices_vitd, default='Diğer')
        
    return df
def generate_stat_table_advanced(df: pd.DataFrame, groups_col: str, params: list, force_parametric: bool = False):
    results = []
    
    # Hangi sütuna göre grupluyorsak, o sütundaki geçerli grupları belirle
    # Grupların sırasını (1., 2., 3. diye numaralandırdığımız için) sort ediyoruz.
    if groups_col not in df.columns:
        return pd.DataFrame()

    # NaN olmayan benzersiz grupları al ve sırala
    valid_groups = sorted([g for g in df[groups_col].unique() if pd.notna(g) and g != 'Diğer'])
    
    # Eğer grup sayısı 2'den azsa istatistik yapılamaz
    if len(valid_groups) < 2:
        return pd.DataFrame()

    # 1. BAŞLIKLARI VE TOPLAM SAYILARI SABİTLE
    group_counts = df[groups_col].value_counts()
    
    # Dinamik başlık listesi oluştur
    col_names = {}
    for g in valid_groups:
        count = group_counts.get(g, 0)
        col_names[g] = f"{g} (n={count})"
    
    df_stat = df[df[groups_col].isin(valid_groups)].copy()

    for p in params:
        if p not in df_stat.columns:
            continue
            
        clean_col = df_stat.dropna(subset=[p])
        
        # Grupları ayır (Dynamic List Comprehension)
        groups_data = [clean_col[clean_col[groups_col] == g][p] for g in valid_groups]
        
        # Her grupta en az 3 veri var mı kontrolü
        if any(len(g) < 3 for g in groups_data):
            continue
            
        # 2. NORMALLİK TESTİ
        is_normal = False
        if not force_parametric:
            try:
                # Tüm gruplar için Shapiro testi
                p_values = [shapiro(g)[1] for g in groups_data]
                is_normal = all(p > 0.05 for p in p_values)
            except:
                is_normal = False 
        
        # 3. FORMATLAMA VE TEST
        row = {"Parametre": p}
        
        if force_parametric or is_normal:
            # --- PARAMETRİK (Mean ± SD) ---
            for g, data in zip(valid_groups, groups_data):
                row[col_names[g]] = f"{data.mean():.2f} ± {data.std():.2f}"
            
            try:
                # Dinamik argüman aktarımı (*) ile ANOVA
                _, p_val = f_oneway(*groups_data)
                test_desc = "ANOVA"
            except:
                p_val = 1.0
                test_desc = "Hata"
        else:
            # --- NON-PARAMETRİK (Median [Min-Max]) ---
            for g, data in zip(valid_groups, groups_data):
                row[col_names[g]] = f"{data.median():.2f} ({data.min():.2f} - {data.max():.2f})"
            
            try:
                # Dinamik argüman aktarımı (*) ile Kruskal-Wallis
                _, p_val = kruskal(*groups_data)
                test_desc = "Kruskal-Wallis"
            except:
                p_val = 1.0
                test_desc = "Hata"

        p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
        row["P Değeri"] = p_text
        row["Metod"] = test_desc
        
        results.append(row)
        
    return pd.DataFrame(results)

def plot_group_comparison(df, group_col, value_col, force_parametric=False):
    """
    Referans görsele benzer şekilde:
    1. Ham veriyi nokta olarak basar (Strip Plot).
    2. %95 Güven Aralığını (CI) ve Ortalamayı/Medyanı çizer (Point Plot).
    3. Global P değerini başlığa yazar.
    """
    # Veri Hazırlığı
    valid_groups = sorted([g for g in df[group_col].unique() if pd.notna(g) and str(g) != 'Diğer'])
    plot_df = df[df[group_col].isin(valid_groups)].copy()
    
    # Boş veri temizliği
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return None

    # Grafik Alanı Oluştur
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Renk Paleti
    palette = sns.color_palette("viridis", n_colors=len(valid_groups))

    # 1. HAM VERİ NOKTALARI (Strip Plot) - Resimdeki dağınık noktalar
    sns.stripplot(
        data=plot_df, x=group_col, y=value_col, 
        order=valid_groups, jitter=0.2, alpha=0.6, size=4, palette=palette, ax=ax, zorder=0
    )

    # 2. %95 GÜVEN ARALIĞI VE MERKEZ (Point Plot) - Resimdeki siyah çizgiler
    # Parametrik -> Ortalama ve %95 CI
    # Non-Parametrik -> Medyan ve %95 CI (Bootstrap ile hesaplanır)
    estimator = np.mean if force_parametric else np.median
    est_label = "Ortalama" if force_parametric else "Medyan"
    
    sns.pointplot(
        data=plot_df, x=group_col, y=value_col,
        order=valid_groups, estimator=estimator, errorbar=('ci', 95),
        color='black', capsize=0.1, join=False, markers="_", scale=0, err_kws={'linewidth': 2}, ax=ax, zorder=10
    )
    
    # Ortaya belirgin bir nokta koy (Merkezi eğilim için)
    sns.pointplot(
        data=plot_df, x=group_col, y=value_col,
        order=valid_groups, estimator=estimator, errorbar=None,
        color='black', join=False, markers="D", scale=0.8, ax=ax, zorder=11
    )

    # 3. İSTATİSTİK TESTİ (Başlık İçin)
    groups_data = [plot_df[plot_df[group_col] == g][value_col] for g in valid_groups]
    
    p_val = 1.0
    test_name = ""
    try:
        if force_parametric:
            _, p_val = f_oneway(*groups_data)
            test_name = "ANOVA"
        else:
            _, p_val = kruskal(*groups_data)
            test_name = "Kruskal-Wallis"
    except:
        pass

    p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
    
    # Grafik Süslemeleri
    ax.set_title(f"{value_col} Dağılımı ve %95 Güven Aralığı ({est_label})\n{test_name} P-Değeri: {p_text}", fontsize=14, fontweight='bold')
    ax.set_xlabel(group_col.replace("_", " "), fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine() # Çerçevenin üst ve sağ çizgisini kaldır (Resimdeki gibi temiz görünüm)

    return fig
    
@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str, encoding: str, user_sep: str):
    ext = os.path.splitext(filename.lower())[1]

    # ---- Excel ----
    if ext in [".xlsx", ".xls"]:
        try:
            bio = BytesIO(file_bytes)
            df = pd.read_excel(bio)
            return df, "excel", None
        except Exception as e:
            raise ValueError(f"Excel dosyası okunamadı. Format hatası: {e}")

    # ---- CSV ----
    text = file_bytes.decode(encoding, errors="replace")

    try:
        sniff = csv.Sniffer().sniff(text[:20000], delimiters=[",", ";", "\t", "|"])
        sep = sniff.delimiter
    except Exception:
        sep = user_sep if user_sep else ";"

    bad_lines = []

    def bad_handler(line):
        bad_lines.append(line)
        return None

    df = pd.read_csv(StringIO(text), sep=sep, engine="python", on_bad_lines=bad_handler)
    return df, f"csv(sep='{sep}')", bad_lines


# -----------------------------
# UI (ARAYÜZ) - SIDEBAR DÜZELTİLMİŞ HALİ
# -----------------------------
st.title("Hemogram -> B12 ve Vitamin D Tahmini (Regresyon)")

with st.sidebar:
    st.header("Veri")
    uploaded = st.file_uploader(
        "Dosya yükle (XLSX / CSV)",
        type=["xlsx", "xls", "csv"]
    )
    sep = st.selectbox("CSV ayırıcı", [",", ";", "\t", "|"], index=1)
    encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "cp1254", "latin1"], index=1)

    st.divider()
    st.header("Model Ayarları")
    seed = st.number_input("Random seed", value=42, step=1)
    test_size = st.slider("Test oranı", 0.1, 0.4, 0.2, 0.05)

    available_models = [
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "HuberRegressor (robust)",
        "RandomForest", "ExtraTrees", "GradientBoosting", "HistGradientBoosting",
        "XGBoost (if installed)", "LightGBM (if installed)", "CatBoost (if installed)"
    ]
    model_name = st.selectbox("Model seç", available_models, index=6)
    scale_numeric = st.checkbox("Sayısal değişkenleri ölçekle", value=False)

    st.divider()
    st.header("Analiz ve İstatistik")
    
    # --- DÜZELTME: Hedef seçimi ve Force Parametric burada tek seferde tanımlanıyor ---
    target_choice = st.radio("Hedef", ["B12", "VİTAMİN D"], index=0)
    
    # Yeni eklediğimiz özellik:
    force_para = st.checkbox("Normallik Testini Yoksay (Hepsini Mean ± SD Ver)", value=False)
    
    do_multitarget = st.checkbox("Çoklu Hedef Raporu (İkisini de analiz et)", value=True)

    st.divider()
    st.header("Değerlendirme")
    cv_folds = st.slider("CV fold", 3, 10, 5, 1)
    do_perm_importance = st.checkbox("Permutation importance hesapla", value=False)
    perm_repeats = st.slider("Permutation tekrar", 2, 10, 5, 1)

st.caption("Not: Bu uygulama klinik karar aracı değildir; araştırma/hipotez amaçlıdır.")


# --- MAIN KISMI ---

# ---------------------------------------------------------
# ADIM 1: DOSYAYI OKU, KONTROL ET VE 'df' OLUŞTUR (ÖNCE BU ÇALIŞMALI)
# ---------------------------------------------------------
if uploaded is None:
    st.info("Başlamak için XLSX veya CSV dosyanı yükle.")
    st.stop()

# Tip kontrolü (net hata mesajı)
ext = os.path.splitext(uploaded.name.lower())[1]
if ext not in [".xlsx", ".xls", ".csv"]:
    st.error(f"Desteklenmeyen dosya türü: {ext} (Sadece .xlsx / .xls / .csv)")
    st.stop()

# Read (XLSX/CSV)
file_bytes = uploaded.getvalue()
try:
    df_raw, read_mode, bad_lines = read_uploaded_file(
        file_bytes=file_bytes,
        filename=uploaded.name,
        encoding=encoding,
        user_sep=sep,
    )
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

st.success(f"Dosya okundu ✅ ({read_mode}) | satır: {len(df_raw):,} | sütun: {df_raw.shape[1]}")

if bad_lines:
    st.warning(f"{len(bad_lines)} bozuk satır CSV'den atlandı.")

# Veriyi Temizle
df = clean_dataframe(df_raw)

# İndeks Hesaplama
df = calculate_derived_indices(df)

# 1. Yaş Grupları
if "HASTA_YAS" in df.columns:
    df = segment_age_groups(df)

# 2. Klinik Gruplar (B12 ve Vit D) - YENİ EKLENDİ
df = segment_clinical_groups(df)

# -----------------------------
# İSTATİSTİK TABLOSU GÖSTERİMİ
# -----------------------------
st.divider()
st.header("📋 Detaylı Klinik İstatistikler")

# Analiz edilecek tüm parametreler (B12 ve Vit D buraya EKLENDİ)
target_params = [
    "B12", "VİTAMİN D", # <-- İsteğiniz üzerine eklendi
    "WBC", "HGB", "HCT", "MCV", "PLT", "NE#", "LY#", "MO#", "EO#", "BA#", 
    "RDW-CV", "RDW-SD", "MPV", "PCT", "PDW",
    "NLR", "PLR", "LMR", "SII", "SIRI", "AISI", "Mentzer"
]
present_params = [p for p in target_params if p in df.columns]

# Kullanıcıya Hangi Gruplamayı İstediğini Sor
group_options = {}
if "Yas_Grubu" in df.columns:
    group_options["Yaş Grupları (Okul Öncesi vs.)"] = "Yas_Grubu"
if "B12_Grubu" in df.columns:
    group_options["B12 Durumu (Düşük/Normal/Yüksek)"] = "B12_Grubu"
if "VitD_Grubu" in df.columns:
    group_options["Vitamin D Durumu (Eksik/Yeterli)"] = "VitD_Grubu"

if group_options:
    selected_label = st.radio("Tablo Gruplama Kriteri Seçiniz:", list(group_options.keys()), horizontal=True)
    selected_group_col = group_options[selected_label]
    
    st.info(f"Aşağıdaki tablo **{selected_label}** kriterine göre oluşturulmuştur.")
    
    # Tabloyu oluştur
    stat_table = generate_stat_table_advanced(df, selected_group_col, present_params, force_parametric=force_para)
    
    if not stat_table.empty:
        st.dataframe(stat_table, use_container_width=True, hide_index=True)
        
        # CSV İndir
        def convert_df(d):
            return d.to_csv(index=False, sep=";").encode('utf-8-sig')
        
        csv_name = f"istatistik_{selected_group_col}.csv"
        st.download_button(
            label="Tabloyu İndir (CSV)",
            data=convert_df(stat_table),
            file_name=csv_name,
            mime="text/csv"
        )
    else:
        st.warning("Seçilen grup için yeterli veri bulunamadı.")
# ... (st.download_button kodunun hemen altı) ...
        
        # --- GRAFİK BÖLÜMÜ (YENİ EKLENEN KISIM) ---
        st.divider()
        st.subheader("📊 Grafiksel Analiz (%95 CI)")
        
        # Hangi parametreyi çizmek istediğini sor
        graph_param = st.selectbox(
            "Grafiğini çizmek istediğiniz parametreyi seçin:",
            options=present_params,
            index=0
        )
        
        if graph_param:
            st.markdown(f"**{graph_param}** parametresinin **{selected_label}** gruplarına göre dağılımı:")
            
            # Grafiği Çiz
            fig = plot_group_comparison(df, selected_group_col, graph_param, force_parametric=force_para)
            
            if fig:
                st.pyplot(fig)
                st.caption(
                    "**Grafik Açıklaması:** Renkli noktalar bireysel hasta verilerini gösterir. "
                    "Siyah kare/çizgi ise grubun **Merkezi Eğilimini (Medyan/Ortalama)** ve **%95 Güven Aralığını (CI)** temsil eder. "
                    "Güven aralıkları örtüşmüyorsa gruplar arası farkın anlamlı olma ihtimali yüksektir."
                )
            else:
                st.warning("Grafik oluşturulamadı (Yetersiz veri).")


else:
    st.warning("Gruplama yapılabilecek veri (Yaş, B12 veya Vit D) bulunamadı.")

st.divider()
st.caption("Not: Bu uygulama klinik karar aracı değildir; araştırma/hipotez amaçlıdır.")

if uploaded is None:
    st.info("Başlamak için XLSX veya CSV dosyanı yükle.")
    st.stop()

# Tip kontrolü (net hata mesajı)
ext = os.path.splitext(uploaded.name.lower())[1]
if ext not in [".xlsx", ".xls", ".csv"]:
    st.error(f"Desteklenmeyen dosya türü: {ext} (Sadece .xlsx / .xls / .csv)")
    st.stop()

# Read (XLSX/CSV)
file_bytes = uploaded.getvalue()
try:
    df_raw, read_mode, bad_lines = read_uploaded_file(
        file_bytes=file_bytes,
        filename=uploaded.name,
        encoding=encoding,
        user_sep=sep,
    )
except Exception as e:
    st.error(f"Dosya okunamadı: {e}")
    st.stop()

st.success(f"Dosya okundu ✅ ({read_mode}) | satır: {len(df_raw):,} | sütun: {df_raw.shape[1]}")

if bad_lines:
    st.warning(f"{len(bad_lines)} bozuk satır CSV'den atlandı. İlk 2 satır:")
    st.code("\n".join([str(x) for x in bad_lines[:2]]))

df = clean_dataframe(df_raw)

# ... (Dosya okuma ve clean_dataframe işlemleri bittikten hemen sonra) ...

# 1. İndeks Hesaplama
df = calculate_derived_indices(df)

# 2. Yaş Gruplama (Sadece 0-17 yaş arası için)
if "HASTA_YAS" in df.columns:
    # Tüm veride hesaplama yapalım ama tabloyu filtreleyelim
    df = segment_age_groups(df)

st.divider()
st.header("📋 Klinik İstatistikler (Otomatik Dağılım Analizi)")
st.info("Her parametre için **Shapiro-Wilk** testi uygulanır. Dağılım normalse **Ortalama ± SS**, değilse **Medyan (Min-Max)** gösterilir.")

if "Yas_Grubu" in df.columns:
    target_params = [
        "WBC", "HGB", "HCT", "MCV", "PLT", "NE#", "LY#", "MO#", "EO#", "BA#", 
        "RDW-CV", "MPV", "NLR", "PLR", "LMR", "SII", "SIRI", "Mentzer"
    ]
    present_params = [p for p in target_params if p in df.columns]
    
    # YENİ FONKSİYONU ÇAĞIRIYORUZ
    stat_table = generate_stat_table_advanced(df, "Yas_Grubu", present_params)
    
    if not stat_table.empty:
        st.dataframe(stat_table, use_container_width=True, hide_index=True)
        
        def convert_df(d):
            return d.to_csv(index=False, sep=";").encode('utf-8-sig')

        st.download_button(
            label="İstatistik Tablosunu İndir (CSV)",
            data=convert_df(stat_table),
            file_name="klinik_istatistik_shapiro.csv",
            mime="text/csv"
        )
    else:
        st.warning("Veri yok veya yaş grupları uygun değil.")
else:
    st.warning("Yaş verisi bulunamadı.")

st.divider()
# ... (Buradan itibaren mevcut ML kodlarınız devam edebilir: st.header("Model") vs.) ...

st.subheader("Veri Önizleme")
st.write(df.head(10))

required_targets = {"B12", "VİTAMİN D"}
missing_targets = required_targets - set(df.columns)
if missing_targets:
    st.error(f"Hedef sütun(lar) eksik: {missing_targets}. Kolon isimlerini kontrol et.")
    st.stop()

groups = get_feature_groups(df.columns.tolist())
group_name = st.selectbox("Özellik seti", list(groups.keys()), index=0)
feature_cols = groups[group_name]

if not feature_cols:
    st.error("Seçilen özellik setinde hiç sütun bulunamadı. Kolon isimlerini kontrol et.")
    st.stop()

df_model = df.dropna(subset=[target_choice]).copy()

if "HASTA_YAS" in df_model.columns:
    df_model = df_model[df_model["HASTA_YAS"].between(0, 16, inclusive="both")]

st.write(f"Modelleme için örnek sayısı: **{len(df_model):,}**")

X = df_model[feature_cols].copy()
for c in X.columns:
    if c != "CINSIYET" and X[c].dtype == object:
        X[c] = safe_to_numeric(X[c])

X_train, X_test, y_train, y_test = train_test_split(
    X, df_model[target_choice], test_size=float(test_size), random_state=int(seed)
)

try:
    reg = build_model(model_name, int(seed))
except Exception as e:
    st.error(str(e))
    st.stop()

pipe = build_pipeline(X_train, reg, scale_numeric=scale_numeric)

st.subheader("Çapraz Doğrulama (CV)")
cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))
scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

with st.spinner("CV hesaplanıyor..."):
    # neg_root_mean_squared_error hala string olarak desteklense de metrics fonksiyonumuz manuel.
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

cv_r2 = float(np.mean(cv_res["test_r2"]))
cv_mae = float(-np.mean(cv_res["test_mae"]))
cv_rmse = float(-np.mean(cv_res["test_rmse"]))

c1, c2, c3 = st.columns(3)
c1.metric("CV R²", f"{cv_r2:.3f}")
c2.metric("CV MAE", f"{cv_mae:.3f}")
c3.metric("CV RMSE", f"{cv_rmse:.3f}")

st.subheader("Test Sonuçları")
with st.spinner("Model eğitiliyor..."):
    pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
m = compute_metrics(y_test, y_pred)

c1, c2, c3 = st.columns(3)
c1.metric("Test R²", f"{m['R2']:.3f}")
c2.metric("Test MAE", f"{m['MAE']:.3f}")
c3.metric("Test RMSE", f"{m['RMSE']:.3f}")

res_df = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred})
res_df["residual"] = res_df["y_true"] - res_df["y_pred"]
st.write(res_df.head(20))

if do_perm_importance:
    st.subheader("Özellik Önemi (Permutation Importance)")
    with st.spinner("Permutation importance hesaplanıyor (biraz sürebilir)..."):
        # YENİ CACHED FONKSİYONU ÇAĞIRIYORUZ
        r = calculate_permutation_importance(pipe, X_test, y_test, int(perm_repeats), int(seed))
        
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std
    }).sort_values("importance_mean", ascending=False)
    st.dataframe(imp, use_container_width=True)

if do_multitarget:
    st.subheader("B12 + Vitamin D (İki hedef ayrı ayrı rapor)")
    targets = ["B12", "VİTAMİN D"]
    report_rows = []
    for t in targets:
        df_tmp = df.dropna(subset=[t]).copy()
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
            "n": int(len(df_tmp)),
            "R2": float(met["R2"]),
            "MAE": float(met["MAE"]),
            "RMSE": float(met["RMSE"]),
        })

    report = pd.DataFrame(report_rows)
    st.dataframe(report, use_container_width=True)

st.success("Bitti ✅")
