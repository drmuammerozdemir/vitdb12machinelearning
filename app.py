# -*- coding: utf-8 -*-
"""
==========================================================================
 PEDİATRİK POPÜLASYONDA B12 ve VİTAMİN D EKSİKLİĞİ – STREAMLIT UYGULAMASI
==========================================================================
 Kullanım:
     1) pip install streamlit pandas numpy scipy scikit-learn matplotlib
                    seaborn openpyxl statsmodels xlsxwriter
     2) streamlit run app.py
==========================================================================
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import (
    shapiro, kstest, ttest_ind, mannwhitneyu, f_oneway, kruskal,
    chi2_contingency, pearsonr, spearmanr,
)
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

st.set_page_config(
    page_title="Pediatrik B12 & Vit D Analizi",
    layout="wide",
    page_icon="🧪",
)

# =========================================================================
# YARDIMCI FONKSİYONLAR
# =========================================================================
@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    # Olası isim varyasyonlarını standartlaştır
    rename_map = {
        "VİTAMİN D": "VITD",  "VITAMIN D": "VITD",  "VIT D": "VITD",
        "VITD": "VITD",
        "CİNSİYET": "CINSIYET", "CINSIYET": "CINSIYET",
        "HASTA_YAS": "YAS", "HASTA YAŞ": "YAS", "YAŞ": "YAS",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def to_numeric_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """İnflamatuar indeksleri hesaplar.

    Matematiksel tanımsızlığı (payda = 0) açıkça NaN ile işaretler;
    epsilon ile maskelemez. Klinik olarak lenfosit/monosit = 0 olduğunda
    indeksin "tanımsız" olduğunu doğru yansıtır — sahte astronomik
    değerler üretmez.
    """
    NE  = pd.to_numeric(df["NE#"], errors="coerce")
    LY  = pd.to_numeric(df["LY#"], errors="coerce")
    MO  = pd.to_numeric(df["MO#"], errors="coerce")
    PLT = pd.to_numeric(df["PLT"], errors="coerce")
    WBC = pd.to_numeric(df["WBC"], errors="coerce")

    # Payda 0 → NaN: bölme matematiksel olarak doğal NaN üretir
    LY_s = LY.replace(0, np.nan)
    MO_s = MO.replace(0, np.nan)
    dWBCNE = (WBC - NE).where((WBC - NE) > 0)   # negatif/0 → NaN
    LY_PLT = (LY * PLT).where((LY * PLT) > 0)

    df["NLR"]  = NE / LY_s
    df["PLR"]  = PLT / LY_s
    df["MLR"]  = MO / LY_s
    df["LMR"]  = LY / MO_s
    df["NMR"]  = NE / MO_s
    df["dNLR"] = NE / dWBCNE
    df["SII"]  = (PLT * NE) / LY_s
    df["SIRI"] = (NE * MO) / LY_s
    df["AISI"] = (NE * MO * PLT) / LY_s
    df["PIV"]  = (NE * MO * PLT) / LY_s  # Literatürde AISI ile aynı formül
    df["NLPR"] = NE / LY_PLT

    # Güvenlik ağı: olası kalan ±inf değerleri NaN'a çevir
    idx_cols = ["NLR","PLR","MLR","LMR","NMR","dNLR","SII","SIRI","AISI","PIV","NLPR"]
    df[idx_cols] = df[idx_cols].replace([np.inf, -np.inf], np.nan)
    return df


def classify_b12(x, cut_def=200, cut_low=300):
    if pd.isna(x): return np.nan
    if x < cut_def: return "Eksik"
    if x < cut_low: return "Sınırda"
    return "Normal"


def classify_vitd(x, cut_def=12, cut_ins=20):
    if pd.isna(x): return np.nan
    if x < cut_def: return "Eksik"
    if x < cut_ins: return "Yetersiz"
    return "Yeterli"


def age_group(y):
    """
    Türk pediatri literatürüne (Neyzi/Ertuğrul, Temel Pediatri) uygun
    yaş gruplaması.
    """
    if pd.isna(y): return np.nan
    if y < 2:   return "1-Süt çocuğu (0-1 y)"
    if y < 6:   return "2-Oyun çocuğu (2-5 y)"
    if y < 12:  return "3-Okul çağı (6-11 y)"
    return "4-Adölesan (12-18 y)"


# =========================================================================
# İLERİ İSTATİSTİK YARDIMCI FONKSİYONLAR
# =========================================================================
def delong_roc_test(y_true, scores_a, scores_b):
    """
    DeLong's nonparametric test for comparing two correlated ROC AUCs.
    Returns: dict with AUCs, difference, z-statistic, p-value, 95% CI.
    """
    from scipy.stats import norm
    y_true = np.asarray(y_true).astype(int)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return None

    def _struct(scores):
        X = np.asarray(scores)[pos][:, None]
        Y = np.asarray(scores)[neg][None, :]
        comp = (X > Y).astype(float) + 0.5*(X == Y).astype(float)
        return comp.mean(axis=1), comp.mean(axis=0)

    V10a, V01a = _struct(scores_a)
    V10b, V01b = _struct(scores_b)
    auc_a, auc_b = V10a.mean(), V10b.mean()

    s10aa = np.var(V10a, ddof=1); s10bb = np.var(V10b, ddof=1)
    s10ab = np.cov(V10a, V10b, ddof=1)[0, 1]
    s01aa = np.var(V01a, ddof=1); s01bb = np.var(V01b, ddof=1)
    s01ab = np.cov(V01a, V01b, ddof=1)[0, 1]

    var_a = s10aa/m + s01aa/n
    var_b = s10bb/m + s01bb/n
    cov_ab = s10ab/m + s01ab/n
    var_diff = var_a + var_b - 2*cov_ab

    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b,
                "diff": auc_a-auc_b, "z": np.nan, "p": np.nan,
                "ci_low": np.nan, "ci_high": np.nan}

    se_diff = np.sqrt(var_diff)
    z = (auc_a - auc_b) / se_diff
    p = 2 * (1 - norm.cdf(abs(z)))
    ci_low = (auc_a - auc_b) - 1.96*se_diff
    ci_high = (auc_a - auc_b) + 1.96*se_diff
    return {"auc_a": auc_a, "auc_b": auc_b,
            "diff": auc_a-auc_b, "z": z, "p": p,
            "ci_low": ci_low, "ci_high": ci_high}


def hosmer_lemeshow(y_true, y_prob, g=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    from scipy.stats import chi2
    d = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                      "p": np.asarray(y_prob).astype(float)})
    try:
        d["bin"] = pd.qcut(d["p"], q=g, duplicates="drop")
    except ValueError:
        return None
    grp = d.groupby("bin", observed=True)
    obs_pos = grp["y"].sum()
    n_grp = grp["y"].count()
    obs_neg = n_grp - obs_pos
    exp_pos = grp["p"].sum()
    exp_neg = n_grp - exp_pos
    valid = (exp_pos > 0) & (exp_neg > 0)
    if valid.sum() < 3:
        return None
    chi = (((obs_pos[valid] - exp_pos[valid])**2 / exp_pos[valid]) +
           ((obs_neg[valid] - exp_neg[valid])**2 / exp_neg[valid])).sum()
    df_hl = max(1, valid.sum() - 2)
    p_val = 1 - chi2.cdf(chi, df_hl)
    return {"chi2": float(chi), "df": int(df_hl), "p": float(p_val),
            "n_bins": int(valid.sum())}


def decision_curve(y_true, y_prob, thresholds):
    """Decision Curve Analysis — net benefit for model vs treat-all vs treat-none."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true); n_event = int(y_true.sum())
    nb_model, nb_all = [], []
    for pt in thresholds:
        if pt >= 1.0:
            nb_model.append(np.nan); nb_all.append(np.nan); continue
        pred = (y_prob >= pt).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        nb_model.append((tp/n) - (fp/n) * (pt/(1-pt)))
        nb_all.append((n_event/n) - ((n - n_event)/n) * (pt/(1-pt)))
    return np.array(nb_model), np.array(nb_all), np.zeros(len(thresholds))


def cliffs_delta(x, y):
    """Cliff's Delta — nonparametric effect size [-1, 1]."""
    x = np.asarray(x); y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    X = x[:, None]; Y = y[None, :]
    cmp = (X > Y).astype(float) - (X < Y).astype(float)
    return float(cmp.mean())


def cohens_d(x, y):
    """Cohen's d — parametric standardized mean difference."""
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return np.nan
    sx2 = np.var(x, ddof=1); sy2 = np.var(y, ddof=1)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx + ny - 2))
    if sp == 0: return np.nan
    return float((np.mean(x) - np.mean(y)) / sp)


def cramers_v(contingency):
    """Cramér's V — categorical effect size [0, 1]."""
    try:
        ct = contingency.values if hasattr(contingency, "values") \
                                else np.asarray(contingency)
        chi2_stat, _, _, _ = chi2_contingency(ct, correction=False)
        n = ct.sum()
        if n == 0: return np.nan
        r, k = ct.shape
        denom = n * (min(r-1, k-1))
        if denom <= 0: return np.nan
        return float(np.sqrt(chi2_stat / denom))
    except Exception:
        return np.nan


def fmt_p(p):
    """p-değerini biçimlendir (publication style)."""
    if pd.isna(p): return "—"
    if p < 0.001: return "< 0.001"
    if p < 0.01:  return f"{p:.3f}"
    return f"{p:.3f}"


# =========================================================================
# MULTIKOLİNEERİTE TESTİ
# =========================================================================
def compute_vif(X_array, var_names):
    """Variance Inflation Factor — VIF > 10: ciddi multikolineerite."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    rows = []
    Xc = sm.add_constant(X_array, has_constant="add")
    for i, name in enumerate(var_names):
        try:
            v = variance_inflation_factor(Xc, i+1)  # +1 → constant atla
        except Exception:
            v = np.nan
        rows.append({
            "Değişken": name,
            "VIF": (round(v, 2) if not pd.isna(v) else "—"),
            "Yorum": (
                "✅ İyi" if (not pd.isna(v) and v < 5)
                else ("⚠️ Orta" if (not pd.isna(v) and v < 10)
                      else ("❌ Yüksek" if (not pd.isna(v) and v < 30)
                            else ("🔴 Çok yüksek" if not pd.isna(v)
                                  else "—")))
            ),
        })
    return pd.DataFrame(rows)


def find_perfect_collinearity(X_array, var_names, thresh=0.999):
    """Mükemmel veya neredeyse mükemmel kolineer çiftleri bul."""
    n = X_array.shape[1]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            try:
                r = np.corrcoef(X_array[:, i], X_array[:, j])[0, 1]
            except Exception:
                continue
            if abs(r) >= thresh:
                pairs.append({"Değişken 1": var_names[i],
                              "Değişken 2": var_names[j],
                              "Korelasyon (r)": round(r, 6)})
    return pd.DataFrame(pairs)


def brier_score(y_true, y_prob):
    """Brier skoru — düşük = iyi kalibrasyon (0 mükemmel, 0.25 rastgele)."""
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    return float(np.mean((y_prob - y_true)**2))


def optimism_corrected_auc(X, y, n_boot=200, penalty="l2",
                           C=1.0, random_state=42):
    """
    Harrell's bootstrap .632 optimism-correction for AUC.

    1. Apparent AUC: tüm veride fit + tüm veride test
    2. Her bootstrap için: bootstrap örneğinde fit
       - bootstrap örneğinde test (optimistik)
       - orijinal veride test (gerçekçi)
       - optimism = optimistik - gerçekçi
    3. Corrected AUC = Apparent - mean(optimism)
    """
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(random_state)
    n = len(y)

    def _fit_auc(Xtr, ytr, Xte, yte):
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return np.nan
        m = LogisticRegression(penalty=penalty, C=C,
                               solver="lbfgs", max_iter=2000)
        m.fit(Xtr, ytr)
        return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    apparent = _fit_auc(X, y, X, y)
    optimisms = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_boot = _fit_auc(Xb, yb, Xb, yb)     # optimistik
        auc_orig = _fit_auc(Xb, yb, X, y)       # gerçekçi
        if not (np.isnan(auc_boot) or np.isnan(auc_orig)):
            optimisms.append(auc_boot - auc_orig)
    if not optimisms:
        return apparent, apparent, 0.0
    mean_opt = float(np.mean(optimisms))
    corrected = apparent - mean_opt
    return apparent, corrected, mean_opt


def build_table_one(df, group_col, group1, group2,
                    num_vars, cat_vars=None, alpha=0.05):
    """
    Publication-ready Table 1 (Springer/Wiley/Lancet style).
    - Sürekli normal: mean ± SD, t-test (³), Cohen's d
    - Sürekli non-normal: median (Q1–Q3), Mann-Whitney U (¹), Cliff's Delta
    - Kategorik: n (%), Chi-square Yates (²), Cramér's V
    """
    g1 = df[df[group_col] == group1].copy()
    g2 = df[df[group_col] == group2].copy()
    n1, n2 = len(g1), len(g2)
    rows = []

    for var in num_vars:
        if var not in df.columns: continue
        x = pd.to_numeric(g1[var], errors="coerce").dropna()
        y = pd.to_numeric(g2[var], errors="coerce").dropna()
        if len(x) < 3 or len(y) < 3: continue

        try:
            _, pnx = shapiro(x.values[:5000])
            _, pny = shapiro(y.values[:5000])
            is_normal = (pnx > alpha) and (pny > alpha)
        except Exception:
            is_normal = False

        if is_normal:
            _, p_val = ttest_ind(x, y, equal_var=False)
            g1_str = f"{x.mean():.2f} ± {x.std(ddof=1):.2f}"
            g2_str = f"{y.mean():.2f} ± {y.std(ddof=1):.2f}"
            es = cohens_d(x, y); es_label = "Cohen's d"
            marker = "³"
        else:
            try:
                _, p_val = mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                p_val = np.nan
            g1_str = (f"{x.median():.2f} "
                      f"({x.quantile(0.25):.2f}–{x.quantile(0.75):.2f})")
            g2_str = (f"{y.median():.2f} "
                      f"({y.quantile(0.25):.2f}–{y.quantile(0.75):.2f})")
            es = cliffs_delta(x.values, y.values); es_label = "Cliff's Δ"
            marker = "¹"

        rows.append({
            "Variable": var,
            f"{group1} (n = {n1})": g1_str,
            f"{group2} (n = {n2})": g2_str,
            "p-value": fmt_p(p_val) + marker,
            "Effect size": (f"{es:.3f}" if not pd.isna(es) else "—"),
            "_es_label": es_label,
        })

    if cat_vars:
        for var in cat_vars:
            if var not in df.columns: continue
            try:
                ct = pd.crosstab(df[group_col], df[var])
                if group1 not in ct.index or group2 not in ct.index:
                    continue
                ct_sub = ct.loc[[group1, group2]]
                if ct_sub.values.sum() == 0: continue
                _, p_val, _, _ = chi2_contingency(
                    ct_sub.values,
                    correction=(ct_sub.shape == (2, 2))
                )
                es = cramers_v(ct_sub); es_label = "Cramér's V"
            except Exception:
                continue

            cats = list(ct_sub.columns)
            cats_label = "/".join(map(str, cats))
            g1_parts = [
                f"{int(ct_sub.loc[group1, c])} "
                f"({100*ct_sub.loc[group1, c]/max(1, n1):.1f}%)"
                for c in cats
            ]
            g2_parts = [
                f"{int(ct_sub.loc[group2, c])} "
                f"({100*ct_sub.loc[group2, c]/max(1, n2):.1f}%)"
                for c in cats
            ]
            rows.append({
                "Variable": f"{var} ({cats_label})",
                f"{group1} (n = {n1})": "/".join(g1_parts),
                f"{group2} (n = {n2})": "/".join(g2_parts),
                "p-value": fmt_p(p_val) + "²",
                "Effect size": (f"{es:.3f}" if not pd.isna(es) else "—"),
                "_es_label": es_label,
            })

    out = pd.DataFrame(rows)
    return out


def descriptive(df, cols):
    rows = []
    for c in cols:
        if c not in df.columns: continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0: continue
        rows.append({
            "Değişken": c,
            "n": len(s),
            "Ort": round(s.mean(), 3),
            "SS": round(s.std(), 3),
            "Medyan": round(s.median(), 3),
            "Q1": round(s.quantile(.25), 3),
            "Q3": round(s.quantile(.75), 3),
            "Min": round(s.min(), 3),
            "Max": round(s.max(), 3),
            "Skewness": round(s.skew(), 3),
            "Kurtosis": round(s.kurt(), 3),
        })
    return pd.DataFrame(rows)


def normality_table(df, cols, n_sample=5000):
    rows = []
    for c in cols:
        if c not in df.columns: continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) < 3:
            rows.append([c, len(x), None, None, "Veri yok", "-"]); continue
        if len(x) <= n_sample:
            stat, p = shapiro(x.sample(min(len(x), n_sample), random_state=42))
            test = "Shapiro-Wilk"
        else:
            z = (x - x.mean()) / x.std(ddof=1)
            stat, p = kstest(z, "norm")
            test = "Kolmogorov-Smirnov"
        rows.append([c, len(x), round(stat, 4), round(p, 6),
                     "Normal" if p > 0.05 else "Normal Değil", test])
    return pd.DataFrame(rows, columns=["Değişken","n","stat","p","Yorum","Test"])


def freq_table(s: pd.Series, name):
    f = s.value_counts(dropna=False).to_frame("n")
    f["%"] = (f["n"] / f["n"].sum() * 100).round(2)
    f.index.name = name
    return f.reset_index()


def two_group_test(df, group_col, val_cols, g1, g2, normal_map=None, correction="fdr_bh"):
    rows = []
    for c in val_cols:
        if c not in df.columns: continue
        a = pd.to_numeric(df.loc[df[group_col] == g1, c], errors="coerce").dropna()
        b = pd.to_numeric(df.loc[df[group_col] == g2, c], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3: continue
        is_normal = normal_map.get(c, False) if normal_map else False
        if is_normal:
            stat, p = ttest_ind(a, b, equal_var=False)
            test_name = "Welch t"
        else:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            test_name = "Mann-Whitney U"
        rows.append({
            "Değişken": c,
            f"{g1} n": len(a),
            f"{g1} medyan [Q1-Q3]":
                f"{a.median():.2f} [{a.quantile(.25):.2f}-{a.quantile(.75):.2f}]",
            f"{g2} n": len(b),
            f"{g2} medyan [Q1-Q3]":
                f"{b.median():.2f} [{b.quantile(.25):.2f}-{b.quantile(.75):.2f}]",
            "stat": round(stat, 3),
            "p": p,
            "Test": test_name,
        })
    out = pd.DataFrame(rows)
    if not out.empty and correction:
        rej, padj, *_ = multipletests(out["p"], method=correction)
        out["p_adj"] = padj
        out["Anlamlı"] = np.where(rej, "✓", "")
    if not out.empty:
        out["p"] = out["p"].round(5)
        out["p_adj"] = out["p_adj"].round(5)
    return out


def multi_group_test(df, group_col, val_cols, normal_map=None, correction="fdr_bh"):
    rows = []
    for c in val_cols:
        if c not in df.columns: continue
        groups = [pd.to_numeric(g[c], errors="coerce").dropna() for _, g in df.groupby(group_col)]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2: continue
        is_normal = normal_map.get(c, False) if normal_map else False
        if is_normal:
            stat, p = f_oneway(*groups); test_name = "ANOVA"
        else:
            stat, p = kruskal(*groups);  test_name = "Kruskal-Wallis"
        rows.append({"Değişken": c, "stat": round(stat, 3), "p": p, "Test": test_name})
    out = pd.DataFrame(rows)
    if not out.empty and correction:
        rej, padj, *_ = multipletests(out["p"], method=correction)
        out["p_adj"] = padj
        out["Anlamlı"] = np.where(rej, "✓", "")
        out["p"] = out["p"].round(5); out["p_adj"] = out["p_adj"].round(5)
    return out


def chi2_with_summary(df, row_col, col_col):
    ct = pd.crosstab(df[row_col], df[col_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ct, None
    chi2, p, dof, exp = chi2_contingency(ct)
    # Cramér's V
    n = ct.values.sum()
    cramer_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    summary = {"chi2": round(chi2, 3), "dof": dof, "p": round(p, 5),
               "Cramér V": round(cramer_v, 3)}
    return ct, summary


def corr_table(df, target, cols, method="spearman"):
    rows = []
    x = pd.to_numeric(df[target], errors="coerce")
    for c in cols:
        if c == target or c not in df.columns: continue
        y = pd.to_numeric(df[c], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() < 5: continue
        if method == "spearman":
            r, p = spearmanr(x[m], y[m])
        else:
            r, p = pearsonr(x[m], y[m])
        rows.append({"Değişken": c, "r": round(r, 3), "p": p, "n": int(m.sum())})
    out = pd.DataFrame(rows).sort_values("p")
    if not out.empty:
        rej, padj, *_ = multipletests(out["p"], method="fdr_bh")
        out["p_adj"] = padj.round(5)
        out["Anlamlı"] = np.where(rej, "✓", "")
        out["p"] = out["p"].round(5)
    return out


def roc_plot(df, group_col, value_col, positive_label):
    sub = df[[group_col, value_col]].dropna()
    y = (sub[group_col] == positive_label).astype(int)
    if y.nunique() < 2: return None, None
    score = sub[value_col].astype(float)
    # Hangi yön daha iyi ayırıyorsa onu kullan
    fpr1, tpr1, thr1 = roc_curve(y,  score);  auc1 = auc(fpr1, tpr1)
    fpr2, tpr2, thr2 = roc_curve(y, -score);  auc2 = auc(fpr2, tpr2)
    if auc1 >= auc2:
        fpr, tpr, thr, AUC, direction = fpr1, tpr1, thr1, auc1, "↑ pozitif"
    else:
        fpr, tpr, thr, AUC, direction = fpr2, tpr2, -thr2, auc2, "↓ pozitif"
    # En iyi cut-off (Youden J)
    J = tpr - fpr
    ix = np.argmax(J)
    return {"fpr": fpr, "tpr": tpr, "thr": thr, "auc": AUC,
            "best_cut": float(thr[ix]),
            "sens": float(tpr[ix]), "spec": float(1 - fpr[ix]),
            "direction": direction,
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}, sub


def to_excel_download(dfs: dict) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as w:
        for name, d in dfs.items():
            if isinstance(d, pd.DataFrame) and not d.empty:
                d.to_excel(w, sheet_name=name[:31], index=False)
    return out.getvalue()


# =========================================================================
# YAN PANEL – VERİ YÜKLEME & PARAMETRELER
# =========================================================================
st.sidebar.title("⚙️ Ayarlar")

uploaded = st.sidebar.file_uploader("Excel dosyanızı yükleyin", type=["xlsx","xls"])
st.sidebar.markdown("---")

st.sidebar.subheader("Eksiklik Eşik Değerleri")
b12_def  = st.sidebar.number_input("B12 eksik <", value=200, step=10)
b12_low  = st.sidebar.number_input("B12 sınırda <", value=300, step=10)
vitd_def = st.sidebar.number_input("Vit D eksik <", value=12, step=1)
vitd_ins = st.sidebar.number_input("Vit D yetersiz <", value=20, step=1)
st.sidebar.caption("Pediatrik eşik (Munns 2016): eksik 12 altı, yetersiz 12-20, yeterli 20 ustu ng/mL")

st.sidebar.subheader("Yaş Filtresi (Pediatri)")
age_min, age_max = st.sidebar.slider("Yaş aralığı (yıl)", 0, 25, (0, 18))

correction = st.sidebar.selectbox(
    "Çoklu karşılaştırma düzeltmesi",
    ["fdr_bh","bonferroni","holm","none"], index=0)
correction = None if correction == "none" else correction

# =========================================================================
# ANA SAYFA
# =========================================================================
st.title("🧪 Pediatrik Popülasyonda B12 & Vitamin D Eksikliği – İleri Analiz")

if uploaded is None:
    st.info("Sol panelden Excel dosyanızı yükleyin. "
            "Dosyada şu sütunlar bulunmalı: BA#, BA%, EO#, EO%, HCT, HGB, "
            "LY#, LY%, MCH, MCHC, MCV, MO#, MO%, MPV, NE#, NE%, PCT, PDW, "
            "PLT, RBC, RDW-CV, RDW-SD, WBC, B12, VİTAMİN D, CINSIYET, HASTA_YAS")
    st.stop()

df = load_excel(uploaded)

NUM_COLS = ["BA#","BA%","EO#","EO%","HCT","HGB","LY#","LY%","MCH","MCHC",
            "MCV","MO#","MO%","MPV","NE#","NE%","PCT","PDW","PLT","RBC",
            "RDW-CV","RDW-SD","WBC","B12","VITD","YAS"]
df = to_numeric_cols(df, NUM_COLS)

# Cinsiyet etiketi
if "CINSIYET" in df.columns:
    df["CINSIYET_LBL"] = df["CINSIYET"].map({1: "Erkek", 2: "Kız"})

# Filtreler
n0 = len(df)
df = df[(df["YAS"].between(age_min, age_max))].copy()
n_after_age = len(df)
df = df[(df["WBC"] > 0) & (df["LY#"] > 0) & (df["NE#"] >= 0) & (df["PLT"] > 0)].copy()
n_after_cbc = len(df)

# İndeksler
df = compute_indices(df)

# Sınıflandırmalar
df["B12_KAT"]  = df["B12"].apply(lambda v: classify_b12(v, b12_def, b12_low))
df["VITD_KAT"] = df["VITD"].apply(lambda v: classify_vitd(v, vitd_def, vitd_ins))
df["B12_EKSIK"]   = (df["B12"]  < b12_def).astype("Int64")
df["VITD_EKSIK"]  = (df["VITD"] < vitd_def).astype("Int64")
df["YAS_GRUBU"]   = df["YAS"].apply(age_group)

# --- STROBE akis sayilari (kohort secim semasi icin) ---
try:
    _b12_ok  = df["B12"].notna()
    _vitd_ok = df["VITD"].notna()
    _cls_mask = _b12_ok & _vitd_ok
    _cls = df[_cls_mask].copy()
    _b = _cls["B12_EKSIK"].astype("Int64").astype(float)
    _d = _cls["VITD_EKSIK"].astype("Int64").astype(float)
    FLOW_COUNTS = {
        "n0": int(n0),
        "n_after_age": int(n_after_age),
        "n_after_cbc": int(n_after_cbc),
        "n_excl_age": int(n0 - n_after_age),
        "n_excl_cbc": int(n_after_age - n_after_cbc),
        "n_missing_lab": int((~_cls_mask).sum()),
        "n_class": int(_cls_mask.sum()),
        "g_kontrol":  int(((_b == 0) & (_d == 0)).sum()),
        "g_izoleD":   int(((_b == 0) & (_d == 1)).sum()),
        "g_izoleB12": int(((_b == 1) & (_d == 0)).sum()),
        "g_kombine":  int(((_b == 1) & (_d == 1)).sum()),
    }
except Exception:
    FLOW_COUNTS = {}

# Genel özet
c1, c2, c3, c4 = st.columns(4)
c1.metric("Toplam hasta", f"{len(df):,}", delta=f"{len(df)-n0}")
if "CINSIYET_LBL" in df.columns:
    c2.metric("Kız", int((df["CINSIYET_LBL"]=="Kız").sum()))
    c3.metric("Erkek", int((df["CINSIYET_LBL"]=="Erkek").sum()))
c4.metric("Ortalama yaş (yıl)", f"{df['YAS'].mean():.2f}")

with st.expander("📋 Verinin İlk 20 Satırı"):
    st.dataframe(df.head(20))

# =========================================================================
# SEKMELER
# =========================================================================
def univariate_logistic(df, y_col, predictors):
    """Her prediktor icin tek-degiskenli lojistik regresyon (OR per 1-SD)."""
    rows = []
    y_all = pd.to_numeric(df[y_col], errors="coerce")
    for p in predictors:
        if p == "CINSIYET":
            if "CINSIYET_LBL" not in df.columns:
                continue
            x_raw = (df["CINSIYET_LBL"] == "Erkek").astype(float)
            is_std = False
        else:
            if p not in df.columns:
                continue
            x_raw = pd.to_numeric(df[p], errors="coerce")
            is_std = True
        d = pd.DataFrame({"y": y_all, "x": x_raw}).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if d["y"].nunique() < 2 or len(d) < 10:
            continue
        xv = d["x"].values.astype(float)
        if is_std and xv.std() > 0:
            xv = (xv - xv.mean()) / xv.std()
        Xc = sm.add_constant(xv, has_constant="add")
        try:
            m = sm.Logit(d["y"].values.astype(int), Xc).fit(disp=0, maxiter=200)
            beta = float(m.params[1]); pval = float(m.pvalues[1])
            lo_b, hi_b = m.conf_int()[1]
            OR = float(np.exp(beta))
            lo = float(np.exp(lo_b)); hi = float(np.exp(hi_b))
        except Exception:
            continue
        rows.append({
            "Değişken": p, "OR": OR, "%95 CI alt": lo, "%95 CI üst": hi,
            "p": pval, "n": int(len(d)),
            "Ölçek": "1-SD" if is_std else "Erkek vs Kız",
        })
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_FDR"] = multipletests(res["p"], method="fdr_bh")[1]
        res = res.sort_values("p").reset_index(drop=True)
    return res


tabs = st.tabs([
    "1) Tanımlayıcı İstatistik",
    "2) Frekans Tabloları",
    "3) Normallik Testleri",
    "4) Grup Karşılaştırmaları",
    "5) Ki-kare",
    "6) Korelasyon",
    "7) ROC Analizi",
    "8) Görselleştirme",
    "9) Regresyon Analizi",
    "10) Tablo 1 (Manuscript)",
    "11) İndir / Rapor",
])

# Analiz değişken listeleri
HEMA_COLS = ["WBC","NE#","LY#","MO#","EO#","BA#","HGB","HCT","MCV","MCH",
             "MCHC","RDW-CV","RDW-SD","PLT","MPV","PCT","PDW","RBC"]
INDEX_COLS = ["NLR","PLR","MLR","LMR","NMR","dNLR","SII","SIRI","AISI","PIV","NLPR"]
LAB_COLS = ["B12","VITD"]
ALL_NUM = [c for c in (["YAS"] + HEMA_COLS + LAB_COLS + INDEX_COLS) if c in df.columns]

# --- Sekme 1 ----------------------------------------------------------------
with tabs[0]:
    with st.expander("🧭 Kohort akış şeması (STROBE) — sayılar", expanded=True):
        fc = FLOW_COUNTS
        if not fc:
            st.info("Akış sayıları hesaplanamadı.")
        else:
            st.caption(
                "Şekildeki n = ___ kutularını bu sayılarla doldurun. Ara bant "
                "(Vit D 12–20 / B12 200–300) dışlanmıyor; yeterli/normal "
                "grubuna dahildir."
            )
            ladder = pd.DataFrame([
                {"Adım": "Veri tabanı kayıtları (ham)",
                 "Çıkarılan": "—", "Kalan n": fc["n0"]},
                {"Adım": f"Yaş {age_min}–{age_max} dışı çıkarıldı",
                 "Çıkarılan": fc["n_excl_age"], "Kalan n": fc["n_after_age"]},
                {"Adım": "Geçersiz CBC (WBC/LY/PLT ≤ 0) çıkarıldı",
                 "Çıkarılan": fc["n_excl_cbc"], "Kalan n": fc["n_after_cbc"]},
                {"Adım": "B12 veya Vit D eksik (sınıflanamadı)",
                 "Çıkarılan": fc["n_missing_lab"], "Kalan n": fc["n_class"]},
            ])
            st.dataframe(ladder, use_container_width=True, hide_index=True)
            nT = max(fc["n_class"], 1)
            groups = pd.DataFrame([
                {"Grup": "Kontrol", "n": fc["g_kontrol"],
                 "%": f"{100*fc['g_kontrol']/nT:.1f}%"},
                {"Grup": "İzole D eksikliği", "n": fc["g_izoleD"],
                 "%": f"{100*fc['g_izoleD']/nT:.1f}%"},
                {"Grup": "İzole B12 eksikliği", "n": fc["g_izoleB12"],
                 "%": f"{100*fc['g_izoleB12']/nT:.1f}%"},
                {"Grup": "Kombine eksiklik", "n": fc["g_kombine"],
                 "%": f"{100*fc['g_kombine']/nT:.1f}%"},
            ])
            st.dataframe(groups, use_container_width=True, hide_index=True)
            st.caption(
                f"Sınıflanabilir kohort: n = {fc['n_class']:,}  |  4 grup toplamı: "
                f"{fc['g_kontrol']+fc['g_izoleD']+fc['g_izoleB12']+fc['g_kombine']:,}"
            )

    st.subheader("Tanımlayıcı İstatistikler")
    desc = descriptive(df, ALL_NUM)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Cinsiyete Göre Tanımlayıcı (Medyan [Q1-Q3])")
    if "CINSIYET_LBL" in df.columns:
        rows = []
        for c in ALL_NUM:
            sub = df.groupby("CINSIYET_LBL")[c].agg(
                lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]")
            rows.append({"Değişken": c, **sub.to_dict()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --- Sekme 2 ----------------------------------------------------------------
with tabs[1]:
    st.subheader("Kategorik Frekanslar")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cinsiyet**")
        if "CINSIYET_LBL" in df.columns:
            st.dataframe(freq_table(df["CINSIYET_LBL"], "Cinsiyet"))
        st.markdown("**Yaş Grubu**")
        st.dataframe(freq_table(df["YAS_GRUBU"], "Yaş Grubu"))
    with col2:
        st.markdown("**B12 Durumu**")
        st.dataframe(freq_table(df["B12_KAT"], "B12"))
        st.markdown("**Vitamin D Durumu**")
        st.dataframe(freq_table(df["VITD_KAT"], "VitD"))

    st.subheader("Çapraz Tablolar")
    if "CINSIYET_LBL" in df.columns:
        st.markdown("**Cinsiyet × B12**")
        st.dataframe(pd.crosstab(df["CINSIYET_LBL"], df["B12_KAT"], margins=True))
        st.markdown("**Cinsiyet × Vit D**")
        st.dataframe(pd.crosstab(df["CINSIYET_LBL"], df["VITD_KAT"], margins=True))
    st.markdown("**Yaş Grubu × B12**")
    st.dataframe(pd.crosstab(df["YAS_GRUBU"], df["B12_KAT"], margins=True))
    st.markdown("**Yaş Grubu × Vit D**")
    st.dataframe(pd.crosstab(df["YAS_GRUBU"], df["VITD_KAT"], margins=True))

# --- Sekme 3 ----------------------------------------------------------------
with tabs[2]:
    st.subheader("Normallik Testleri")
    st.caption("n ≤ 5000 → Shapiro-Wilk, n > 5000 → Kolmogorov-Smirnov")
    norm_tbl = normality_table(df, ALL_NUM)
    st.dataframe(norm_tbl, use_container_width=True)
    normal_map = dict(zip(norm_tbl["Değişken"], norm_tbl["Yorum"] == "Normal"))

# --- Sekme 4 ----------------------------------------------------------------
with tabs[3]:
    st.subheader("İki Grup Karşılaştırmaları")
    target = st.selectbox(
        "Karşılaştırma grubu",
        ["B12_KAT (Eksik vs Normal)", "VITD_KAT (Eksik vs Yeterli)",
         "B12_EKSIK (1 vs 0)", "VITD_EKSIK (1 vs 0)", "CINSIYET_LBL (Erkek vs Kız)"]
    )
    norm_map = dict(zip(normality_table(df, ALL_NUM)["Değişken"],
                        normality_table(df, ALL_NUM)["Yorum"] == "Normal"))

    if target.startswith("B12_KAT"):
        out = two_group_test(df, "B12_KAT", ALL_NUM, "Eksik", "Normal", norm_map, correction)
    elif target.startswith("VITD_KAT"):
        out = two_group_test(df, "VITD_KAT", ALL_NUM, "Eksik", "Yeterli", norm_map, correction)
    elif target.startswith("B12_EKSIK"):
        out = two_group_test(df, "B12_EKSIK", ALL_NUM, 1, 0, norm_map, correction)
    elif target.startswith("VITD_EKSIK"):
        out = two_group_test(df, "VITD_EKSIK", ALL_NUM, 1, 0, norm_map, correction)
    else:
        out = two_group_test(df, "CINSIYET_LBL", ALL_NUM, "Erkek", "Kız", norm_map, correction)
    st.dataframe(out, use_container_width=True)

    st.subheader("Çok Grup Karşılaştırması (Yaş Grubu / B12 Kategorisi / VitD Kategorisi)")
    group_col = st.radio("Grup değişkeni", ["YAS_GRUBU","B12_KAT","VITD_KAT"], horizontal=True)
    mout = multi_group_test(df, group_col, ALL_NUM, norm_map, correction)
    st.dataframe(mout, use_container_width=True)

# --- Sekme 5 ----------------------------------------------------------------
with tabs[4]:
    st.subheader("Ki-Kare Testleri & Cramér V")
    pairs = [("CINSIYET_LBL","B12_KAT"),
             ("CINSIYET_LBL","VITD_KAT"),
             ("YAS_GRUBU","B12_KAT"),
             ("YAS_GRUBU","VITD_KAT"),
             ("B12_KAT","VITD_KAT")]
    for r, c in pairs:
        if r not in df.columns or c not in df.columns: continue
        ct, summ = chi2_with_summary(df, r, c)
        st.markdown(f"**{r} × {c}**")
        st.dataframe(ct)
        if summ: st.write(summ)
        st.markdown("---")

# --- Sekme 6 ----------------------------------------------------------------
with tabs[5]:
    st.subheader("Korelasyon Analizi (B12 ve Vit D ile)")
    method = st.radio("Yöntem", ["spearman","pearson"], horizontal=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**B12 ile korelasyonlar**")
        st.dataframe(corr_table(df, "B12", ALL_NUM, method), use_container_width=True)
    with colB:
        st.markdown("**Vit D ile korelasyonlar**")
        st.dataframe(corr_table(df, "VITD", ALL_NUM, method), use_container_width=True)

    st.subheader("Korelasyon Isı Haritası")
    pick_cols = st.multiselect("Değişkenleri seç", ALL_NUM,
        default=["B12","VITD","NLR","PLR","MLR","dNLR","SII","SIRI","PIV","HGB","WBC","PLT"])
    if len(pick_cols) >= 2:
        cmat = df[pick_cols].corr(method=method)
        fig, ax = plt.subplots(figsize=(min(1+0.6*len(pick_cols),16),
                                        min(1+0.5*len(pick_cols),12)))
        sns.heatmap(cmat, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink":.7})
        ax.set_title(f"{method.title()} Korelasyon Matrisi")
        st.pyplot(fig)

# --- Sekme 7 ----------------------------------------------------------------
with tabs[6]:
    st.subheader("ROC Analizi – Eksikliği Tahmin Etmede İndekslerin Performansı")
    outcome = st.selectbox("Bağımlı değişken (pozitif sınıf = eksik)",
                           ["B12_EKSIK","VITD_EKSIK"])
    pred = st.selectbox("Yordayıcı (tarama testi)",
                        INDEX_COLS + ["WBC","HGB","HCT","MCV","RDW-CV","PLT","MPV"])
    res, sub = roc_plot(df, outcome, pred, 1)
    if res is None:
        st.warning("Seçilen değişkenlerle ROC hesaplanamadı.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{res['auc']:.3f}")
        c2.metric("Best cut-off", f"{res['best_cut']:.3f}")
        c3.metric("Sensitivite", f"{res['sens']:.3f}")
        c4.metric("Spesifite",  f"{res['spec']:.3f}")
        st.caption(f"Yön: {res['direction']} · pozitif n={res['n_pos']} · negatif n={res['n_neg']}")
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(res["fpr"], res["tpr"], lw=2, label=f"{pred} (AUC={res['auc']:.3f})")
        ax.plot([0,1],[0,1], "k--", alpha=0.5)
        ax.set_xlabel("1 - Spesifite"); ax.set_ylabel("Sensitivite")
        ax.set_title(f"ROC – {outcome} ~ {pred}"); ax.legend()
        st.pyplot(fig)

    st.subheader("Tüm İndeksler için Toplu AUC Tablosu")
    rows = []
    for c in INDEX_COLS:
        r, _ = roc_plot(df, outcome, c, 1)
        if r: rows.append({"Değişken": c, "AUC": round(r["auc"],3),
                           "Cut-off": round(r["best_cut"],3),
                           "Sens": round(r["sens"],3),
                           "Spec": round(r["spec"],3),
                           "Yön": r["direction"]})
    st.dataframe(pd.DataFrame(rows).sort_values("AUC", ascending=False),
                 use_container_width=True)

# --- Sekme 8 ----------------------------------------------------------------
with tabs[7]:
    st.subheader("Dağılım Grafikleri")
    var_to_plot = st.selectbox("Değişken seç", ALL_NUM, index=ALL_NUM.index("B12") if "B12" in ALL_NUM else 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[var_to_plot].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"{var_to_plot} – Histogram")
    sns.boxplot(x="CINSIYET_LBL", y=var_to_plot, data=df, ax=axes[1])
    axes[1].set_title(f"{var_to_plot} – Cinsiyete göre")
    st.pyplot(fig)

    st.subheader("Yaş Grubuna Göre Kutu Grafikleri")
    var2 = st.selectbox("Değişken (yaş grubu için)", ALL_NUM,
                        index=ALL_NUM.index("VITD") if "VITD" in ALL_NUM else 0,
                        key="vbox2")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    order = sorted(df["YAS_GRUBU"].dropna().unique())
    sns.boxplot(x="YAS_GRUBU", y=var2, data=df, order=order, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha="right")
    ax2.set_title(f"{var2} – Yaş gruplarına göre")
    st.pyplot(fig2)

    st.subheader("B12 / Vit D Kategorilerine Göre İndeksler")
    cat_col = st.radio("Kategori", ["B12_KAT","VITD_KAT"], horizontal=True)
    idx_choose = st.selectbox("İndeks", INDEX_COLS, key="idxbox")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.boxplot(x=cat_col, y=idx_choose, data=df, ax=ax3, showfliers=False)
    ax3.set_title(f"{idx_choose} – {cat_col}")
    st.pyplot(fig3)

    # ========================================================================
    # YAYINA HAZIR KARŞILAŞTIRMALI KUTU GRAFİKLERİ
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Yayına Hazır Karşılaştırmalı Kutu Grafikleri")
    st.caption(
        "Birden fazla değişkeni iki grup arasında yan yana karşılaştırın. "
        "Mann-Whitney U testi p-değeri ve anlamlılık yıldızı otomatik olarak eklenir. "
        "300 DPI PNG ve vektörel PDF olarak dergi sunumu için indirilebilir."
    )

    # --- Grup ve değişken seçimi ---
    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        pub_group_col = st.selectbox(
            "Grup değişkeni",
            ["B12_KAT", "VITD_KAT", "CINSIYET_LBL", "YAS_GRUBU"],
            key="pub_grp_col"
        )
    with pcol2:
        pub_palette = st.selectbox(
            "Renk paleti",
            ["Klasik mavi-pembe", "Nature stili", "Lancet stili",
             "Gri tonları (tek renkli baskı)"],
            key="pub_palette"
        )

    palette_map = {
        "Klasik mavi-pembe":            ("#B8D8E3", "#F5B7B1"),
        "Nature stili":                  ("#4E79A7", "#E15759"),
        "Lancet stili":                  ("#00468B", "#ED0000"),
        "Gri tonları (tek renkli baskı)":("#CCCCCC", "#666666"),
    }
    color1, color2 = palette_map[pub_palette]

    grp_options = sorted(df[pub_group_col].dropna().unique().tolist())
    if len(grp_options) < 2:
        st.warning(f"'{pub_group_col}' içinde en az iki grup bulunamadı.")
    else:
        # Mantıklı varsayılan seçimler
        if "Eksik" in grp_options:
            default_g1 = "Eksik"
        else:
            default_g1 = grp_options[0]
        if "Normal" in grp_options:
            default_g2 = "Normal"
        elif "Yeterli" in grp_options:
            default_g2 = "Yeterli"
        else:
            default_g2 = grp_options[-1]

        g1c, g2c = st.columns(2)
        with g1c:
            pub_g1 = st.selectbox("Grup 1 (sol kutu)", grp_options,
                                  index=grp_options.index(default_g1),
                                  key="pub_g1")
        with g2c:
            pub_g2 = st.selectbox("Grup 2 (sağ kutu)", grp_options,
                                  index=grp_options.index(default_g2),
                                  key="pub_g2")

        # Etiket kişiselleştirme (resimdeki "Control / Acne" gibi)
        l1c, l2c = st.columns(2)
        with l1c:
            label_g1 = st.text_input("Grup 1 ekran etiketi",
                                     value=str(pub_g1), key="pub_lbl1")
        with l2c:
            label_g2 = st.text_input("Grup 2 ekran etiketi",
                                     value=str(pub_g2), key="pub_lbl2")

        # Değişken seçimi
        default_vars = [v for v in ["NLR", "SII", "SIRI", "AISI"]
                        if v in ALL_NUM][:2]
        pub_vars = st.multiselect(
            "Karşılaştırılacak değişkenler (1-6 adet önerilir)",
            options=ALL_NUM,
            default=default_vars,
            key="pub_vars",
            max_selections=6
        )

        # İnce ayar
        opt_c1, opt_c2, opt_c3 = st.columns(3)
        with opt_c1:
            show_outliers = st.checkbox("Outlier göster", True, key="pub_out")
        with opt_c2:
            show_n = st.checkbox("n değerlerini göster", False, key="pub_n")
        with opt_c3:
            y_from_zero = st.checkbox("Y-ekseni 0'dan başlasın", True,
                                      key="pub_y0")

        if st.button("📈 Grafikleri Oluştur", type="primary", key="pub_gen"):
            if not pub_vars:
                st.warning("Lütfen en az bir değişken seçin.")
            elif pub_g1 == pub_g2:
                st.warning("Lütfen iki farklı grup seçin.")
            else:
                n_vars = len(pub_vars)
                n_cols = min(3, n_vars)
                n_rows = int(np.ceil(n_vars / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(5.0*n_cols, 5.5*n_rows))
                axes = np.array([axes]).flatten() if n_vars == 1 \
                       else np.array(axes).flatten()

                for i, var in enumerate(pub_vars):
                    ax = axes[i]
                    sub = df[df[pub_group_col].isin([pub_g1, pub_g2])][
                              [pub_group_col, var]].dropna()
                    if sub.empty:
                        ax.text(0.5, 0.5, f"{var}\nveri yok",
                                ha="center", va="center",
                                transform=ax.transAxes)
                        ax.set_axis_off()
                        continue

                    g1_data = sub[sub[pub_group_col] == pub_g1][var]\
                              .astype(float).values
                    g2_data = sub[sub[pub_group_col] == pub_g2][var]\
                              .astype(float).values

                    # Mann-Whitney U
                    if len(g1_data) < 2 or len(g2_data) < 2:
                        p_val = np.nan
                        p_str = "n yetersiz"
                        star = ""
                    else:
                        _, p_val = mannwhitneyu(g1_data, g2_data,
                                                alternative="two-sided")
                        if p_val < 0.001:
                            p_str = "p < 0.001"
                        else:
                            p_str = f"p = {p_val:.3f}"
                        if p_val < 0.001:   star = "***"
                        elif p_val < 0.01:  star = "**"
                        elif p_val < 0.05:  star = "*"
                        else:               star = "ns"

                    # Kutu grafiği
                    bp = ax.boxplot(
                        [g1_data, g2_data], positions=[1, 2], widths=0.55,
                        patch_artist=True, showfliers=show_outliers,
                        medianprops=dict(color="black", linewidth=1.8),
                        whiskerprops=dict(color="black", linewidth=1.0),
                        capprops=dict(color="black", linewidth=1.0),
                        boxprops=dict(linewidth=1.0),
                        flierprops=dict(marker="o", markersize=4,
                                        markerfacecolor="#333333",
                                        markeredgecolor="#333333",
                                        alpha=0.85),
                    )
                    bp["boxes"][0].set_facecolor(color1)
                    bp["boxes"][0].set_edgecolor("#222222")
                    bp["boxes"][1].set_facecolor(color2)
                    bp["boxes"][1].set_edgecolor("#222222")

                    # Anlamlılık çubuğu ve metni
                    all_vals = np.concatenate([g1_data, g2_data])
                    ymax = float(np.nanmax(all_vals))
                    ymin = float(np.nanmin(all_vals))
                    yr = max(ymax - ymin, 1e-9)
                    y_line = ymax + yr*0.08
                    ax.plot([1, 2], [y_line, y_line],
                            color="black", lw=1.3)
                    ax.text(1.5, y_line + yr*0.025,
                            f"{star} {p_str}",
                            ha="center", va="bottom",
                            fontsize=12, style="italic")

                    # Y-ekseni
                    y_low = 0 if y_from_zero else ymin - yr*0.08
                    ax.set_ylim(y_low, y_line + yr*0.25)
                    ax.set_xticks([1, 2])

                    # Etiketler (n ile veya n'siz)
                    if show_n:
                        xl1 = f"{label_g1}\n(n = {len(g1_data)})"
                        xl2 = f"{label_g2}\n(n = {len(g2_data)})"
                    else:
                        xl1, xl2 = label_g1, label_g2
                    ax.set_xticklabels([xl1, xl2], fontsize=11)
                    ax.set_ylabel(var, fontsize=13, fontweight="bold")
                    ax.tick_params(axis="y", labelsize=10)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(axis="y", alpha=0.25,
                            linestyle="-", linewidth=0.5)
                    ax.set_axisbelow(True)

                # Boş subplotları kapat
                for j in range(n_vars, len(axes)):
                    axes[j].set_axis_off()

                plt.tight_layout()
                st.pyplot(fig)

                # PNG (raster, 300 DPI)
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", dpi=300,
                            bbox_inches="tight", facecolor="white")
                # PDF (vektörel, dergi tercihi)
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format="pdf",
                            bbox_inches="tight", facecolor="white")
                # TIFF (bazı dergiler ister)
                buf_tif = io.BytesIO()
                fig.savefig(buf_tif, format="tiff", dpi=300,
                            bbox_inches="tight", facecolor="white")

                dl1, dl2, dl3 = st.columns(3)
                fname = f"boxplots_{pub_group_col}_{pub_g1}_vs_{pub_g2}"
                with dl1:
                    st.download_button(
                        "📥 PNG (300 dpi)",
                        data=buf_png.getvalue(),
                        file_name=f"{fname}.png", mime="image/png",
                        key="dl_png_pub")
                with dl2:
                    st.download_button(
                        "📄 PDF (vektörel)",
                        data=buf_pdf.getvalue(),
                        file_name=f"{fname}.pdf",
                        mime="application/pdf", key="dl_pdf_pub")
                with dl3:
                    st.download_button(
                        "🖼️ TIFF (300 dpi)",
                        data=buf_tif.getvalue(),
                        file_name=f"{fname}.tiff", mime="image/tiff",
                        key="dl_tif_pub")

                # Şekil altı caption önerisi
                caption_lines = [
                    f"**Figure caption önerisi:**",
                    f"*Comparison of "
                    f"{', '.join(pub_vars[:-1]) + (' and ' + pub_vars[-1] if len(pub_vars) > 1 else pub_vars[0])} "
                    f"between {label_g1} (n = {len(df[df[pub_group_col]==pub_g1])}) and "
                    f"{label_g2} (n = {len(df[df[pub_group_col]==pub_g2])}) groups. "
                    f"Boxes represent interquartile range (IQR); horizontal lines indicate medians; "
                    f"whiskers extend to 1.5 × IQR; dots represent outliers. "
                    f"P-values were calculated using the Mann-Whitney U test. "
                    f"*p < 0.05, **p < 0.01, ***p < 0.001; ns: not significant.*"
                ]
                st.info("\n\n".join(caption_lines))

# --- Sekme 9: REGRESYON ANALİZİ ---------------------------------------------
with tabs[8]:
    st.subheader("📈 Lojistik Regresyon ve Regülarize Modeller")
    st.markdown(
        """
        İnflamatuar indekslerin vitamin eksikliğini öngörme gücü çoklu değişkenli
        analizle değerlendirilir. Üç tamamlayıcı yaklaşım:
        - **Standart Lojistik Regresyon** — OR (95% CI) ve p-değerleri
        - **LASSO (L1)** — otomatik değişken seçimi, gereksiz prediktörleri 0'a çeker
        - **Ridge (L2)** — multikolineeriteyi yönetir, katsayıları büzer
        - **Elastic Net** — L1 + L2 hibrit
        """
    )

    # --- Outcome seçimi ---
    o1, o2 = st.columns([1.4, 1])
    with o1:
        outcome_choice = st.selectbox(
            "Çıktı (Outcome) değişkeni",
            ["B12 Eksikliği (<200)", "Vitamin D Eksikliği (<20)",
             "Kombine Eksiklik (B12↓ + D↓)"],
            key="reg_outcome"
        )

    if outcome_choice.startswith("B12"):
        y_col = "B12_EKSIK"; outcome_label = "B12 Deficiency"
    elif outcome_choice.startswith("Vitamin D"):
        y_col = "VITD_EKSIK"; outcome_label = "Vitamin D Deficiency"
    else:
        df["KOMBINE_EKSIK"] = (
            (df.get("B12_EKSIK", 0) == 1) &
            (df.get("VITD_EKSIK", 0) == 1)
        ).astype(int)
        y_col = "KOMBINE_EKSIK"; outcome_label = "Combined Deficiency"

    if y_col not in df.columns:
        st.error(f"'{y_col}' sütunu oluşturulamadı. B12 ve VITD sütunları "
                 f"yüklü mü kontrol edin.")
    else:
        n_pos = int((df[y_col] == 1).sum())
        n_neg = int((df[y_col] == 0).sum())
        with o2:
            st.metric("Pozitif / Negatif",
                      f"{n_pos} / {n_neg}",
                      f"{100*n_pos/(n_pos+n_neg):.1f}% olay")

        # --- Predictor seçimi ---
        candidate_predictors = ([p for p in INDEX_COLS if p in df.columns] +
                                [p for p in HEMA_COLS if p in df.columns])
        if "YAS" in df.columns:
            candidate_predictors.append("YAS")
        if "CINSIYET_LBL" in df.columns:
            candidate_predictors.append("CINSIYET")

        # ============================================================
        # ADIM 1 - UNIVARIATE TARAMA (her degisken tek tek)
        # ============================================================
        with st.expander("Adım 1 — Univariate tarama (tüm değişkenler tek tek)",
                         expanded=True):
            st.caption(
                "Her bağımsız değişken için ayrı tek-değişkenli lojistik "
                "regresyon (sürekli değişkenlerde OR = 1-SD artış başına). "
                "Eşiğin altındakileri tek tıkla multivariate modele aktarın."
            )
            uc1, uc2, uc3 = st.columns([2, 1, 1])
            with uc1:
                univ_vars = st.multiselect(
                    "Taranacak değişkenler",
                    options=candidate_predictors,
                    default=candidate_predictors,
                    key=f"univ_vars_{y_col}",
                )
            with uc2:
                univ_thresh = st.number_input(
                    "p eşiği", value=0.05, min_value=0.001, max_value=0.50,
                    step=0.01, format="%.3f", key=f"univ_thr_{y_col}",
                )
            with uc3:
                univ_ptype = st.radio(
                    "Eşik p türü", ["ham p", "FDR p"], index=0,
                    key=f"univ_pt_{y_col}",
                )
            st.caption(
                "Tarama için literatürde p < 0.20–0.25 de yaygındır "
                "(confounder kaybetmemek için); daha kısıtlı model için 0.05."
            )

            if st.button("Univariate taramayı çalıştır",
                         key=f"run_univ_{y_col}"):
                if not univ_vars:
                    st.warning("En az bir değişken seçin.")
                else:
                    with st.spinner("Univariate modeller çalışıyor..."):
                        st.session_state[f"univ_res_{y_col}"] = \
                            univariate_logistic(df, y_col, univ_vars)

            ures = st.session_state.get(f"univ_res_{y_col}")
            if ures is not None and not ures.empty:
                pcol = "p" if univ_ptype == "ham p" else "p_FDR"
                sig_list = ures.loc[ures[pcol] < univ_thresh,
                                    "Değişken"].tolist()
                disp = ures.copy()
                disp["Anlamlı"] = (ures[pcol] < univ_thresh).map(
                    {True: "evet", False: ""})
                for c in ["OR", "%95 CI alt", "%95 CI üst"]:
                    disp[c] = disp[c].map(lambda v: f"{v:.3f}")
                disp["p"] = disp["p"].map(lambda v: f"{v:.4g}")
                disp["p_FDR"] = disp["p_FDR"].map(lambda v: f"{v:.4g}")
                disp = disp[["Değişken", "OR", "%95 CI alt", "%95 CI üst",
                             "p", "p_FDR", "n", "Ölçek", "Anlamlı"]]
                st.dataframe(disp, use_container_width=True, hide_index=True)
                st.success(
                    f"{pcol} < {univ_thresh:g}: {len(sig_list)} değişken — "
                    + (", ".join(sig_list) if sig_list else "yok")
                )
                if st.button("Anlamlıları multivariate modele aktar",
                             key=f"push_univ_{y_col}", type="primary",
                             disabled=(len(sig_list) == 0)):
                    st.session_state["reg_preds"] = sig_list
                    st.rerun()

        # --- Multivariate prediktor secimi ---
        if "reg_preds" not in st.session_state:
            st.session_state["reg_preds"] = [
                p for p in ["NLR", "SII", "SIRI", "AISI", "YAS"]
                if p in candidate_predictors]
        st.session_state["reg_preds"] = [
            p for p in st.session_state["reg_preds"]
            if p in candidate_predictors]
        reg_predictors = st.multiselect(
            "Bağımsız değişkenler (Predictors) — multivariate model",
            options=candidate_predictors,
            key="reg_preds",
        )

        # --- Model & seçenekler ---
        mc1, mc2 = st.columns(2)
        with mc1:
            models_to_run = st.multiselect(
                "Çalıştırılacak modeller",
                ["Standart Lojistik", "LASSO (L1)",
                 "Ridge (L2)", "Elastic Net"],
                default=["Standart Lojistik", "LASSO (L1)", "Ridge (L2)"],
                key="reg_models"
            )
        with mc2:
            cv_folds = st.slider("Cross-validation k-fold", 3, 10, 5,
                                 key="reg_cv")
            n_bootstrap = st.slider(
                "Bootstrap iterasyon (AUC 95% CI)",
                100, 2000, 500, step=100, key="reg_boot"
            )

        # Regülarizasyon aralığı kontrolü (büyük n için kritik)
        with st.expander("⚙️ İleri ayarlar — Regülarizasyon gücü (λ aralığı)"):
            st.caption(
                "Büyük örneklemlerde (n > 1000) CV otomatik olarak en zayıf "
                "regülarizasyonu seçer; bu durumda LASSO/Ridge standart "
                "lojistikle birleşir. Aralığı **daraltarak** gerçek değişken "
                "seçimi davranışını gözlemleyebilirsiniz."
            )
            lam_c1, lam_c2 = st.columns(2)
            with lam_c1:
                log_C_min = st.slider(
                    "log₁₀(C) alt sınır (C = 1/λ; daha küçük = daha güçlü λ)",
                    -6, 2, -3, key="logCmin",
                )
            with lam_c2:
                log_C_max = st.slider(
                    "log₁₀(C) üst sınır",
                    -2, 4, 1, key="logCmax",
                )
            n_Cs = st.slider("CV nokta sayısı (Cs)", 10, 40, 20, key="nCs")
            check_collin = st.checkbox(
                "🔬 Multikolineerite kontrolü yap (VIF + perfect-r)", True,
                key="reg_vif",
            )

        if st.button("🚀 Regresyon Analizini Çalıştır",
                     type="primary", key="run_reg"):
            if not reg_predictors:
                st.warning("Lütfen en az bir bağımsız değişken seçin.")
            elif not models_to_run:
                st.warning("Lütfen en az bir model seçin.")
            elif n_pos < 10:
                st.warning(f"Pozitif olay sayısı çok düşük (n={n_pos}). "
                           f"Regresyon analizi için minimum ~10×p önerilir.")
            else:
                # ---- Veri hazırlama ----
                base_cols = [p for p in reg_predictors if p != "CINSIYET"]
                cols_needed = base_cols + [y_col]
                if "CINSIYET" in reg_predictors and "CINSIYET_LBL" in df.columns:
                    cols_needed.append("CINSIYET_LBL")
                sub = df[cols_needed].dropna().copy()

                X_cols = list(base_cols)
                if "CINSIYET" in reg_predictors and "CINSIYET_LBL" in sub.columns:
                    sub["CINSIYET_E"] = (sub["CINSIYET_LBL"] == "Erkek").astype(int)
                    X_cols.append("CINSIYET_E")

                X_raw = sub[X_cols].values.astype(float)
                y = sub[y_col].astype(int).values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_raw)

                st.success(f"✅ Analiz edilen örneklem: n = {len(y)} "
                           f"(pozitif: {(y==1).sum()}, negatif: {(y==0).sum()})")

                # ============ MULTİKOLİNEERİTE TANILAMASI ============
                if check_collin:
                    st.markdown("---")
                    st.markdown("### 🔬 Multikolineerite Tanılaması")
                    cond_num = np.linalg.cond(X_raw)
                    perfect_pairs = find_perfect_collinearity(
                        X_raw, X_cols, thresh=0.999
                    )

                    cm1, cm2 = st.columns([1, 2])
                    with cm1:
                        st.metric(
                            "Condition Number",
                            f"{cond_num:.2e}",
                            ("⚠️ > 30" if cond_num > 30
                             else "✅ İyi"),
                            delta_color=("inverse" if cond_num > 30
                                         else "normal"),
                        )

                    if not perfect_pairs.empty:
                        with cm2:
                            st.error(
                                "🔴 **Mükemmel kolineer çiftler bulundu!** "
                                "Singular Matrix hatasının kaynağı bunlardır. "
                                "Bir tanesini çıkarmalısınız:"
                            )
                            st.dataframe(perfect_pairs,
                                         use_container_width=True,
                                         hide_index=True)

                        # Otomatik olarak ikincisini çıkar
                        drop_set = set()
                        for _, r in perfect_pairs.iterrows():
                            if r["Değişken 1"] not in drop_set:
                                drop_set.add(r["Değişken 2"])
                        if drop_set:
                            keep_idx = [i for i, c in enumerate(X_cols)
                                        if c not in drop_set]
                            removed = [c for c in X_cols if c in drop_set]
                            st.info(
                                f"♻️ Otomatik düzeltme: {', '.join(removed)} "
                                f"değişken(ler)i model setinden çıkarıldı "
                                f"(eşdeğer çiftleri korundu)."
                            )
                            X_cols = [X_cols[i] for i in keep_idx]
                            X_raw = X_raw[:, keep_idx]
                            X_scaled = X_scaled[:, keep_idx]

                    # VIF tablosu
                    try:
                        vif_df = compute_vif(X_raw, X_cols)
                        st.markdown("**Variance Inflation Factors (VIF)**")
                        st.dataframe(vif_df,
                                     use_container_width=True,
                                     hide_index=True)
                        n_high = sum(1 for v in vif_df["VIF"]
                                     if isinstance(v, (int, float))
                                     and v >= 10)
                        if n_high > 0:
                            st.warning(
                                f"⚠️ {n_high} değişkende VIF ≥ 10 "
                                f"(orta-yüksek multikolineerite). "
                                f"Bu, standart lojistik regresyonun katsayı "
                                f"varyanslarını şişirir; LASSO/Ridge bu "
                                f"durumda tercih edilir."
                            )
                        else:
                            st.success(
                                "✅ Tüm VIF değerleri kabul edilebilir "
                                "aralıkta."
                            )
                    except Exception as e:
                        st.warning(f"VIF hesaplanamadı: {e}")

                roc_data = {}
                Cs_vals = np.logspace(log_C_min, log_C_max, n_Cs)

                # ============ 1) STANDART LOJİSTİK ============
                if "Standart Lojistik" in models_to_run:
                    fit_ok = False; logit_m = None
                    # Önce Newton, sonra BFGS, sonra LBFGS dene
                    for solver_method in ["newton", "bfgs", "lbfgs"]:
                        try:
                            X_sm = sm.add_constant(X_raw)
                            logit_m = sm.Logit(y, X_sm).fit(
                                disp=0, maxiter=500,
                                method=solver_method,
                            )
                            fit_ok = True
                            if solver_method != "newton":
                                st.info(
                                    f"ℹ️ Standart Lojistik '{solver_method}' "
                                    f"solver ile fit edildi (Newton "
                                    f"singular matrix verdi)."
                                )
                            break
                        except (np.linalg.LinAlgError, Exception):
                            continue

                    if not fit_ok or logit_m is None:
                        st.error(
                            "❌ Standart Lojistik fit edilemedi. "
                            "Tüm solver'lar başarısız oldu. "
                            "Lütfen yüksek-VIF değişkenleri çıkarın "
                            "veya yalnızca regülarize modelleri kullanın."
                        )
                    else:
                        try:
                            params = logit_m.params[1:]
                            ci = logit_m.conf_int()[1:]
                            ors = np.exp(params)
                            ci_low = np.exp(ci[:, 0])
                            ci_high = np.exp(ci[:, 1])
                            pvals = logit_m.pvalues[1:]

                            coef_df = pd.DataFrame({
                                "Değişken": X_cols,
                                "Beta": np.round(params, 4),
                                "OR": np.round(ors, 3),
                                "95% CI Alt": np.round(ci_low, 3),
                                "95% CI Üst": np.round(ci_high, 3),
                                "p": [f"{p:.4f}" for p in pvals],
                            })
                            st.markdown(
                                "### 📋 Standart Lojistik — Odds Ratio Tablosu"
                            )
                            st.dataframe(coef_df,
                                         use_container_width=True)

                            y_pred = logit_m.predict(X_sm)
                            fpr, tpr, _ = roc_curve(y, y_pred)
                            roc_data["Standart Lojistik"] = (
                                fpr, tpr, auc(fpr, tpr), y_pred
                            )
                        except Exception as e:
                            st.error(f"Standart lojistik hatası: {e}")

                # ============ 2) LASSO ============
                if "LASSO (L1)" in models_to_run:
                    try:
                        lasso_cv = LogisticRegressionCV(
                            Cs=Cs_vals, cv=cv_folds, penalty="l1",
                            solver="saga", max_iter=5000,
                            scoring="roc_auc", random_state=42,
                        )
                        lasso_cv.fit(X_scaled, y)
                        coefs = lasso_cv.coef_[0]
                        sel = [(X_cols[i], coefs[i])
                               for i in range(len(X_cols))
                               if abs(coefs[i]) > 1e-6]

                        st.markdown("### 🎯 LASSO (L1) — Seçilen Değişkenler")
                        st.write(f"**Optimal C (1/λ):** {lasso_cv.C_[0]:.4f}  |  "
                                 f"**Seçilen değişken sayısı:** {len(sel)}/{len(X_cols)}")
                        if sel:
                            lasso_df = pd.DataFrame({
                                "Değişken": [s[0] for s in sel],
                                "Beta (standardize)": np.round([s[1] for s in sel], 4),
                                "exp(Beta) (yön)": np.round(
                                    np.exp([s[1] for s in sel]), 3),
                            }).sort_values("Beta (standardize)",
                                           key=abs, ascending=False)
                            st.dataframe(lasso_df, use_container_width=True)
                        else:
                            st.warning("LASSO hiçbir değişken seçmedi — "
                                       "lambda çok yüksek olmuş olabilir.")

                        y_proba = lasso_cv.predict_proba(X_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y, y_proba)
                        roc_data["LASSO (L1)"] = (
                            fpr, tpr, auc(fpr, tpr), y_proba
                        )
                    except Exception as e:
                        st.error(f"LASSO hatası: {e}")

                # ============ 3) RIDGE ============
                if "Ridge (L2)" in models_to_run:
                    try:
                        ridge_cv = LogisticRegressionCV(
                            Cs=Cs_vals, cv=cv_folds, penalty="l2",
                            solver="lbfgs", max_iter=5000,
                            scoring="roc_auc", random_state=42,
                        )
                        ridge_cv.fit(X_scaled, y)
                        coefs = ridge_cv.coef_[0]

                        st.markdown("### 🌐 Ridge (L2) — Katsayılar")
                        st.write(f"**Optimal C (1/λ):** {ridge_cv.C_[0]:.4f}")
                        ridge_df = pd.DataFrame({
                            "Değişken": X_cols,
                            "Beta (standardize)": np.round(coefs, 4),
                            "exp(Beta) (yön)": np.round(np.exp(coefs), 3),
                        }).sort_values("Beta (standardize)",
                                       key=abs, ascending=False)
                        st.dataframe(ridge_df, use_container_width=True)

                        y_proba = ridge_cv.predict_proba(X_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y, y_proba)
                        roc_data["Ridge (L2)"] = (
                            fpr, tpr, auc(fpr, tpr), y_proba
                        )
                    except Exception as e:
                        st.error(f"Ridge hatası: {e}")

                # ============ 4) ELASTIC NET ============
                if "Elastic Net" in models_to_run:
                    try:
                        enet_cv = LogisticRegressionCV(
                            Cs=Cs_vals, cv=cv_folds, penalty="elasticnet",
                            solver="saga", l1_ratios=[0.25, 0.5, 0.75],
                            max_iter=5000, scoring="roc_auc",
                            random_state=42,
                        )
                        enet_cv.fit(X_scaled, y)
                        coefs = enet_cv.coef_[0]

                        st.markdown("### 🧬 Elastic Net — Katsayılar")
                        st.write(f"**Optimal C:** {enet_cv.C_[0]:.4f}  |  "
                                 f"**l1_ratio:** {enet_cv.l1_ratio_[0]:.2f}")
                        enet_df = pd.DataFrame({
                            "Değişken": X_cols,
                            "Beta (standardize)": np.round(coefs, 4),
                        }).sort_values("Beta (standardize)",
                                       key=abs, ascending=False)
                        st.dataframe(enet_df, use_container_width=True)

                        y_proba = enet_cv.predict_proba(X_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y, y_proba)
                        roc_data["Elastic Net"] = (
                            fpr, tpr, auc(fpr, tpr), y_proba
                        )
                    except Exception as e:
                        st.error(f"Elastic Net hatası: {e}")

                # ============ ROC EĞRİSİ KARŞILAŞTIRMASI ============
                if roc_data:
                    st.markdown("---")
                    st.markdown("### 📊 Model Karşılaştırması — ROC Eğrileri")

                    # Bootstrap AUC 95% CI
                    def _boot_ci(y_true, y_pred, n_boot=500):
                        rng = np.random.RandomState(42)
                        aucs = []
                        n = len(y_true)
                        for _ in range(n_boot):
                            idx = rng.randint(0, n, n)
                            if len(np.unique(y_true[idx])) < 2:
                                continue
                            try:
                                aucs.append(
                                    roc_auc_score(y_true[idx], y_pred[idx])
                                )
                            except Exception:
                                continue
                        if not aucs:
                            return (np.nan, np.nan)
                        return tuple(np.percentile(aucs, [2.5, 97.5]))

                    fig_roc, ax = plt.subplots(figsize=(8.5, 7))
                    palette = ["#2E5BFF", "#E74C3C", "#27AE60", "#F39C12"]

                    auc_summary = []
                    for i, (mname, (fpr, tpr, a_val, y_pr)) \
                            in enumerate(roc_data.items()):
                        lo, hi = _boot_ci(y, y_pr, n_bootstrap)
                        color = palette[i % len(palette)]
                        ax.plot(
                            fpr, tpr, color=color, linewidth=2.3,
                            label=(f"{mname}, AUC={a_val:.3f} "
                                   f"(95% CI: {lo:.3f}-{hi:.3f})")
                        )
                        auc_summary.append({
                            "Model": mname,
                            "AUC": round(a_val, 4),
                            "95% CI Alt": round(lo, 4),
                            "95% CI Üst": round(hi, 4),
                        })

                    ax.plot([0, 1], [0, 1], color="black",
                            linestyle="--", linewidth=1.3)
                    ax.set_xlim([-0.02, 1.0])
                    ax.set_ylim([0, 1.02])
                    ax.set_xlabel("1 - Specificity (False Positive Rate)",
                                  fontsize=13)
                    ax.set_ylabel("Sensitivity (True Positive Rate)",
                                  fontsize=13)
                    ax.set_title(
                        f"ROC Curves for Predicting {outcome_label}",
                        fontsize=14, fontstyle="italic", fontweight="bold",
                        pad=15,
                    )
                    ax.legend(
                        loc="lower right", fontsize=10,
                        frameon=True, fancybox=False, edgecolor="black",
                    )
                    ax.grid(alpha=0.25, linestyle="-", linewidth=0.5)
                    ax.set_axisbelow(True)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_roc)

                    # AUC özet tablosu
                    st.markdown("### 📐 AUC Özet Tablosu")
                    st.dataframe(pd.DataFrame(auc_summary),
                                 use_container_width=True)

                    # ====== OPTIMISM-CORRECTED AUC (Bootstrap .632) ======
                    st.markdown("### 🔁 Optimism-Corrected AUC (İç Validasyon)")
                    st.caption(
                        "Harrell bootstrap yöntemiyle iç validasyon. "
                        "Apparent AUC modelin kendi verisindeki performansıdır "
                        "(iyimser); optimism-corrected AUC overfitting'i "
                        "düzelten gerçekçi tahmindir."
                    )
                    with st.spinner("Bootstrap iç validasyon çalışıyor..."):
                        oc_rows = []
                        # Tüm modeller aynı X_scaled'ı kullanır; penalty farkı
                        # optimism üzerinde küçük etki yapar — standart L2 ile
                        # tek bir gerçekçi tahmin sunuyoruz (büyük n'de yeterli).
                        try:
                            app, corr, opt = optimism_corrected_auc(
                                X_scaled, y,
                                n_boot=min(n_bootstrap, 300),
                                penalty="l2", C=1.0,
                            )
                            oc_rows.append({
                                "Yöntem": "Çok değişkenli model (L2)",
                                "Apparent AUC": round(app, 4),
                                "Optimism": round(opt, 4),
                                "Corrected AUC": round(corr, 4),
                            })
                        except Exception as e:
                            st.warning(f"Optimism düzeltmesi hesaplanamadı: {e}")

                    if oc_rows:
                        st.dataframe(pd.DataFrame(oc_rows),
                                     use_container_width=True)
                        oc = oc_rows[0]
                        if oc["Optimism"] < 0.02:
                            st.success(
                                f"✅ Optimism = {oc['Optimism']:.4f} "
                                f"(< 0.02) → model overfitting göstermiyor. "
                                f"Büyük örneklem sayesinde apparent ve "
                                f"corrected AUC neredeyse aynı."
                            )
                        else:
                            st.info(
                                f"ℹ️ Optimism = {oc['Optimism']:.4f}. "
                                f"Corrected AUC'yi raporlamanız önerilir."
                            )


                    # İndirme
                    buf_png = io.BytesIO()
                    fig_roc.savefig(buf_png, format="png", dpi=300,
                                    bbox_inches="tight", facecolor="white")
                    buf_pdf = io.BytesIO()
                    fig_roc.savefig(buf_pdf, format="pdf",
                                    bbox_inches="tight", facecolor="white")
                    buf_tiff = io.BytesIO()
                    fig_roc.savefig(buf_tiff, format="tiff", dpi=300,
                                    bbox_inches="tight", facecolor="white")

                    rd1, rd2, rd3 = st.columns(3)
                    safe_label = outcome_label.replace(" ", "_")
                    with rd1:
                        st.download_button(
                            "📥 PNG (300 dpi)",
                            data=buf_png.getvalue(),
                            file_name=f"roc_comparison_{safe_label}.png",
                            mime="image/png", key="dl_roc_p",
                        )
                    with rd2:
                        st.download_button(
                            "📄 PDF (vektörel)",
                            data=buf_pdf.getvalue(),
                            file_name=f"roc_comparison_{safe_label}.pdf",
                            mime="application/pdf", key="dl_roc_d",
                        )
                    with rd3:
                        st.download_button(
                            "🖼️ TIFF (300 dpi)",
                            data=buf_tiff.getvalue(),
                            file_name=f"roc_comparison_{safe_label}.tiff",
                            mime="image/tiff", key="dl_roc_t",
                        )

                    # Figure caption önerisi
                    n_models = len(roc_data)
                    cap = (
                        f"**Figure caption önerisi:**\n\n"
                        f"*Receiver Operating Characteristic (ROC) curves "
                        f"comparing {n_models} multivariable models for "
                        f"predicting {outcome_label.lower()} from "
                        f"hemogram-derived inflammatory indices. "
                        f"Area Under the Curve (AUC) values with 95% "
                        f"confidence intervals were estimated using "
                        f"{n_bootstrap} bootstrap resamples. "
                        f"Predictors included: "
                        f"{', '.join(X_cols)}. "
                        f"The diagonal dashed line represents the line of "
                        f"no discrimination (AUC = 0.5).*"
                    )
                    st.info(cap)

                # ============ DELONG TESTİ — Çiftli AUC Karşılaştırma ============
                if len(roc_data) >= 2:
                    st.markdown("---")
                    st.markdown("### 🔬 DeLong Testi — Çiftli Model AUC Karşılaştırması")
                    st.caption(
                        "İki modelin AUC'leri istatistiksel olarak farklı mı? "
                        "DeLong (1988) nonparametrik testi, "
                        "korelasyonlu AUC karşılaştırması için altın standarttır."
                    )
                    model_names = list(roc_data.keys())
                    delong_rows = []
                    for i in range(len(model_names)):
                        for j in range(i+1, len(model_names)):
                            m_a, m_b = model_names[i], model_names[j]
                            pa = roc_data[m_a][3]
                            pb = roc_data[m_b][3]
                            res = delong_roc_test(y, pa, pb)
                            if res is None: continue
                            delong_rows.append({
                                "Karşılaştırma": f"{m_a} vs {m_b}",
                                f"AUC ({m_a})": round(res["auc_a"], 4),
                                f"AUC ({m_b})": round(res["auc_b"], 4),
                                "Fark (ΔAUC)": round(res["diff"], 4),
                                "95% CI": (f"[{res['ci_low']:.4f}, "
                                           f"{res['ci_high']:.4f}]"),
                                "z": round(res["z"], 3),
                                "p (DeLong)": fmt_p(res["p"]),
                            })
                    if delong_rows:
                        # Genel kolon adlandırması (her çiftin AUC kolonları farklı)
                        # Bu yüzden basitleştirelim
                        clean_rows = []
                        for r in delong_rows:
                            simple = {
                                "Karşılaştırma": r["Karşılaştırma"],
                                "ΔAUC": r["Fark (ΔAUC)"],
                                "95% CI": r["95% CI"],
                                "z": r["z"],
                                "p (DeLong)": r["p (DeLong)"],
                            }
                            clean_rows.append(simple)
                        st.dataframe(pd.DataFrame(clean_rows),
                                     use_container_width=True)
                        st.caption(
                            "💡 *ΔAUC'nin %95 CI'si 0'ı içermiyorsa veya "
                            "p < 0.05 ise, iki modelin ayırt edici gücü "
                            "istatistiksel olarak farklıdır.*"
                        )

                # ============ KALİBRASYON EĞRİSİ ============
                if roc_data:
                    st.markdown("---")
                    st.markdown("### 🎯 Kalibrasyon (Calibration) Eğrisi & Hosmer-Lemeshow")
                    st.caption(
                        "Tahmin edilen olasılıkların gerçek olay sıklığıyla "
                        "ne kadar uyumlu olduğu. Mükemmel kalibrasyon = "
                        "diyagonal çizgi."
                    )
                    from sklearn.calibration import calibration_curve
                    fig_cal, ax_cal = plt.subplots(figsize=(7.5, 6.5))
                    palette = ["#2E5BFF", "#E74C3C", "#27AE60", "#F39C12"]

                    hl_rows = []
                    for i, (mname, (_, _, _, y_pr)) in enumerate(roc_data.items()):
                        try:
                            frac_pos, mean_pred = calibration_curve(
                                y, y_pr, n_bins=10, strategy="quantile"
                            )
                        except Exception:
                            continue
                        color = palette[i % len(palette)]
                        ax_cal.plot(mean_pred, frac_pos, marker="o",
                                    linewidth=2, color=color,
                                    label=mname, markersize=7)

                        # Hosmer-Lemeshow + Brier
                        hl = hosmer_lemeshow(y, y_pr, g=10)
                        bs = brier_score(y, y_pr)
                        if hl is not None:
                            hl_rows.append({
                                "Model": mname,
                                "Brier Skoru": round(bs, 4),
                                "HL χ²": round(hl["chi2"], 3),
                                "df": hl["df"],
                                "p (HL)": fmt_p(hl["p"]),
                                "Yorum": ("İyi kalibrasyon"
                                          if hl["p"] >= 0.05
                                          else "Zayıf kalibrasyon"),
                            })

                    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.3,
                                label="Mükemmel kalibrasyon")
                    ax_cal.set_xlabel("Predicted Probability",
                                      fontsize=12)
                    ax_cal.set_ylabel("Observed Frequency",
                                      fontsize=12)
                    ax_cal.set_title(
                        f"Calibration Plot — {outcome_label}",
                        fontsize=13, fontstyle="italic", pad=12,
                    )
                    ax_cal.legend(loc="upper left", fontsize=10,
                                  frameon=True, edgecolor="black")
                    ax_cal.grid(alpha=0.25, linestyle="-", linewidth=0.5)
                    ax_cal.set_axisbelow(True)
                    ax_cal.spines["top"].set_visible(False)
                    ax_cal.spines["right"].set_visible(False)
                    ax_cal.set_xlim([-0.02, 1.02])
                    ax_cal.set_ylim([-0.02, 1.02])
                    plt.tight_layout()
                    st.pyplot(fig_cal)

                    if hl_rows:
                        st.markdown(
                            "**Kalibrasyon Metrikleri "
                            "(Brier Skoru + Hosmer-Lemeshow)**"
                        )
                        st.dataframe(pd.DataFrame(hl_rows),
                                     use_container_width=True)
                        st.caption(
                            "💡 *Brier Skoru: 0'a yakın = iyi kalibrasyon "
                            "(0.25 = rastgele tahmin). "
                            "Hosmer-Lemeshow p > 0.05 → model tahminleri ile "
                            "gerçek olay sıklığı arasında anlamlı sapma yok.*"
                        )

                    # Kalibrasyon indirme
                    bc1 = io.BytesIO()
                    fig_cal.savefig(bc1, format="png", dpi=300,
                                    bbox_inches="tight", facecolor="white")
                    bc2 = io.BytesIO()
                    fig_cal.savefig(bc2, format="pdf",
                                    bbox_inches="tight", facecolor="white")
                    cl1, cl2 = st.columns(2)
                    with cl1:
                        st.download_button(
                            "📥 Kalibrasyon PNG",
                            data=bc1.getvalue(),
                            file_name=f"calibration_{outcome_label.replace(' ','_')}.png",
                            mime="image/png", key="dl_cal_png",
                        )
                    with cl2:
                        st.download_button(
                            "📄 Kalibrasyon PDF",
                            data=bc2.getvalue(),
                            file_name=f"calibration_{outcome_label.replace(' ','_')}.pdf",
                            mime="application/pdf", key="dl_cal_pdf",
                        )

                # ============ DECISION CURVE ANALYSIS ============
                if roc_data:
                    st.markdown("---")
                    st.markdown("### 💰 Decision Curve Analysis (DCA)")
                    st.caption(
                        "Modelin **klinik faydası** (net benefit) farklı "
                        "tedavi eşiklerinde değerlendirilir. "
                        "(Vickers & Elkin, 2006)"
                    )
                    thresholds = np.arange(0.01, 0.81, 0.01)
                    fig_dca, ax_dca = plt.subplots(figsize=(8.5, 6.5))
                    palette2 = ["#2E5BFF", "#E74C3C", "#27AE60", "#F39C12"]

                    for i, (mname, (_, _, _, y_pr)) in enumerate(roc_data.items()):
                        nb_m, nb_a, nb_n = decision_curve(y, y_pr, thresholds)
                        color = palette2[i % len(palette2)]
                        ax_dca.plot(thresholds, nb_m,
                                    linewidth=2.2, color=color,
                                    label=f"{mname}")
                        if i == 0:
                            ax_dca.plot(thresholds, nb_a,
                                        linewidth=1.4, color="#888888",
                                        linestyle="--",
                                        label="Treat all")
                            ax_dca.plot(thresholds, nb_n,
                                        linewidth=1.4, color="black",
                                        linestyle=":",
                                        label="Treat none")

                    ax_dca.set_xlabel("Threshold Probability (pt)",
                                      fontsize=12)
                    ax_dca.set_ylabel("Net Benefit", fontsize=12)
                    ax_dca.set_title(
                        f"Decision Curve Analysis — {outcome_label}",
                        fontsize=13, fontstyle="italic", pad=12,
                    )
                    ax_dca.legend(loc="upper right", fontsize=10,
                                  frameon=True, edgecolor="black")
                    ax_dca.grid(alpha=0.25, linestyle="-", linewidth=0.5)
                    ax_dca.set_axisbelow(True)
                    ax_dca.spines["top"].set_visible(False)
                    ax_dca.spines["right"].set_visible(False)
                    # Y eksenini otomatik sınırla
                    all_nbs = []
                    for mname, (_,_,_,y_pr) in roc_data.items():
                        nb_m, _, _ = decision_curve(y, y_pr, thresholds)
                        all_nbs.extend(nb_m[~np.isnan(nb_m)].tolist())
                    if all_nbs:
                        ymin = min(min(all_nbs), -0.05)
                        ymax = max(all_nbs) * 1.1
                        ax_dca.set_ylim([ymin, ymax])
                    plt.tight_layout()
                    st.pyplot(fig_dca)
                    st.caption(
                        "💡 *Eğri 'Treat all' ve 'Treat none' çizgilerinin "
                        "üzerinde kaldığı eşik aralığında modeli kullanmak "
                        "klinik fayda sağlar.*"
                    )

                    # DCA indirme
                    bd1 = io.BytesIO()
                    fig_dca.savefig(bd1, format="png", dpi=300,
                                    bbox_inches="tight", facecolor="white")
                    bd2 = io.BytesIO()
                    fig_dca.savefig(bd2, format="pdf",
                                    bbox_inches="tight", facecolor="white")
                    dca1, dca2 = st.columns(2)
                    with dca1:
                        st.download_button(
                            "📥 DCA PNG",
                            data=bd1.getvalue(),
                            file_name=f"dca_{outcome_label.replace(' ','_')}.png",
                            mime="image/png", key="dl_dca_png",
                        )
                    with dca2:
                        st.download_button(
                            "📄 DCA PDF",
                            data=bd2.getvalue(),
                            file_name=f"dca_{outcome_label.replace(' ','_')}.pdf",
                            mime="application/pdf", key="dl_dca_pdf",
                        )

                # ============ YAŞ GRUBU SUBGROUP ANALİZİ ============
                if "YAS_GRUBU" in df.columns:
                    st.markdown("---")
                    st.markdown("### 👶🧒 Yaş Grubuna Göre Subgroup Analizi")
                    st.caption(
                        "Her gelişim dönemi için ayrı çok değişkenli model. "
                        "Etkinin yaş grupları arasında nasıl değiştiğini "
                        "(effect modification) gösterir. Forest plot ile "
                        "her grubun AUC'si ve %95 CI'si karşılaştırılır."
                    )

                    # Subgroup verisini orijinal df'den, seçili predictorlarla kur
                    sg_cols_needed = X_cols.copy()
                    # X_cols CINSIYET_E içerebilir; orijinal df'de CINSIYET_LBL var
                    sg_base = [c for c in X_cols if c != "CINSIYET_E"]
                    sg_src = df[df[y_col].notna() & df["YAS_GRUBU"].notna()].copy()
                    if "CINSIYET_E" in X_cols and "CINSIYET_LBL" in sg_src.columns:
                        sg_src["CINSIYET_E"] = (
                            sg_src["CINSIYET_LBL"] == "Erkek"
                        ).astype(int)

                    age_groups = sorted(sg_src["YAS_GRUBU"].dropna().unique())
                    sg_rows = []
                    sg_forest = []   # (label, auc, lo, hi, n, events)

                    for ag in age_groups:
                        g_data = sg_src[sg_src["YAS_GRUBU"] == ag]
                        sub_X = g_data[X_cols].apply(
                            pd.to_numeric, errors="coerce")
                        sub_df = pd.concat(
                            [sub_X, g_data[y_col]], axis=1).dropna()
                        if len(sub_df) < 30:
                            sg_rows.append({
                                "Yaş Grubu": ag, "n": len(sub_df),
                                "Olay (n)": "—", "AUC": "yetersiz n",
                                "95% CI": "—",
                            })
                            continue
                        yy = sub_df[y_col].astype(int).values
                        XX = sub_df[X_cols].values.astype(float)
                        n_ev = int((yy == 1).sum())
                        if n_ev < 10 or len(np.unique(yy)) < 2:
                            sg_rows.append({
                                "Yaş Grubu": ag, "n": len(sub_df),
                                "Olay (n)": n_ev, "AUC": "yetersiz olay",
                                "95% CI": "—",
                            })
                            continue
                        try:
                            sc_sg = StandardScaler()
                            XXs = sc_sg.fit_transform(XX)
                            m_sg = LogisticRegression(
                                penalty="l2", C=1.0,
                                solver="lbfgs", max_iter=2000)
                            m_sg.fit(XXs, yy)
                            pr_sg = m_sg.predict_proba(XXs)[:, 1]
                            auc_sg = roc_auc_score(yy, pr_sg)
                            # Bootstrap CI
                            rng = np.random.RandomState(42)
                            boots = []
                            for _ in range(300):
                                idx = rng.randint(0, len(yy), len(yy))
                                if len(np.unique(yy[idx])) < 2:
                                    continue
                                boots.append(
                                    roc_auc_score(yy[idx], pr_sg[idx]))
                            lo, hi = (np.percentile(boots, [2.5, 97.5])
                                      if boots else (np.nan, np.nan))
                            sg_rows.append({
                                "Yaş Grubu": ag, "n": len(sub_df),
                                "Olay (n)": n_ev,
                                "AUC": round(auc_sg, 3),
                                "95% CI": f"[{lo:.3f}, {hi:.3f}]",
                            })
                            sg_forest.append(
                                (ag, auc_sg, lo, hi, len(sub_df), n_ev))
                        except Exception as e:
                            sg_rows.append({
                                "Yaş Grubu": ag, "n": len(sub_df),
                                "Olay (n)": n_ev, "AUC": f"hata",
                                "95% CI": "—",
                            })

                    if sg_rows:
                        st.dataframe(pd.DataFrame(sg_rows),
                                     use_container_width=True,
                                     hide_index=True)

                    # ---- Forest plot ----
                    if len(sg_forest) >= 1:
                        fig_fp, ax_fp = plt.subplots(
                            figsize=(8.5, max(2.5, 0.8*len(sg_forest)+1.5)))
                        ypos = np.arange(len(sg_forest))[::-1]
                        for k, (lab, a_val, lo, hi, nn, ne) \
                                in enumerate(sg_forest):
                            yp = ypos[k]
                            ax_fp.plot([lo, hi], [yp, yp],
                                       color="#2E5BFF", lw=2.2, zorder=1)
                            ax_fp.plot(a_val, yp, "o", color="#E74C3C",
                                       markersize=9, zorder=2)
                            ax_fp.text(hi + 0.01, yp,
                                       f"{a_val:.3f} [{lo:.3f}–{hi:.3f}]  "
                                       f"(n={nn}, olay={ne})",
                                       va="center", fontsize=9.5)
                        ax_fp.axvline(0.5, color="#888888",
                                      linestyle="--", lw=1.2,
                                      label="AUC = 0.5 (rastgele)")
                        ax_fp.set_yticks(ypos)
                        ax_fp.set_yticklabels(
                            [f[0] for f in sg_forest], fontsize=10)
                        ax_fp.set_xlabel("AUC (95% CI)", fontsize=12)
                        ax_fp.set_title(
                            f"Subgroup Analysis by Age — {outcome_label}",
                            fontsize=13, fontstyle="italic", pad=12)
                        ax_fp.set_xlim([0.4, 1.18])
                        ax_fp.legend(loc="lower right", fontsize=9)
                        ax_fp.grid(axis="x", alpha=0.25)
                        ax_fp.spines["top"].set_visible(False)
                        ax_fp.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig_fp)

                        # İndirme
                        bf1 = io.BytesIO()
                        fig_fp.savefig(bf1, format="png", dpi=300,
                                       bbox_inches="tight",
                                       facecolor="white")
                        bf2 = io.BytesIO()
                        fig_fp.savefig(bf2, format="pdf",
                                       bbox_inches="tight",
                                       facecolor="white")
                        fpc1, fpc2 = st.columns(2)
                        with fpc1:
                            st.download_button(
                                "📥 Forest Plot PNG",
                                data=bf1.getvalue(),
                                file_name=f"subgroup_age_{outcome_label.replace(' ','_')}.png",
                                mime="image/png", key="dl_fp_png")
                        with fpc2:
                            st.download_button(
                                "📄 Forest Plot PDF",
                                data=bf2.getvalue(),
                                file_name=f"subgroup_age_{outcome_label.replace(' ','_')}.pdf",
                                mime="application/pdf", key="dl_fp_pdf")

                        st.info(
                            "💡 **Yorum:** Bir yaş grubunda AUC belirgin "
                            "yüksek/düşükse, inflamatuar indekslerin "
                            "öngörü gücü o gelişim döneminde farklı demektir "
                            "(effect modification). CI'ler örtüşmüyorsa "
                            "fark istatistiksel olarak anlamlı olabilir."
                        )

# --- Sekme 10: TABLO 1 (Manuscript) -----------------------------------------
with tabs[9]:
    st.subheader("📋 Tablo 1 — Demografik, Klinik ve Laboratuvar Karşılaştırma")
    st.markdown(
        """
        **Springer / Wiley / Lancet uyumlu, yayına hazır karşılaştırma tablosu.**
        Sürekli değişkenler için normallik durumuna göre otomatik test seçimi
        ve uygun effect size hesaplaması yapılır. Word'e doğrudan
        kopyalanabilir veya HTML / Excel olarak indirilebilir.
        """
    )

    # --- Grup seçimi ---
    tt1, tt2 = st.columns([1.3, 1])
    with tt1:
        t1_group_col = st.selectbox(
            "Grup değişkeni",
            [c for c in ["B12_KAT", "VITD_KAT",
                          "CINSIYET_LBL", "YAS_GRUBU"]
             if c in df.columns],
            key="t1_grp",
        )
    grp_opts = sorted(df[t1_group_col].dropna().unique().tolist())
    if len(grp_opts) < 2:
        st.warning("Bu değişkende en az 2 grup yok.")
    else:
        with tt2:
            st.metric("Toplam grup", f"{len(grp_opts)}")
        # Varsayılan akıllı seçim
        if "Eksik" in grp_opts:
            d_g1 = "Eksik"
        else:
            d_g1 = grp_opts[0]
        d_g2 = ("Normal" if "Normal" in grp_opts
                else ("Yeterli" if "Yeterli" in grp_opts
                      else grp_opts[-1]))

        gc1, gc2 = st.columns(2)
        with gc1:
            t1_g1 = st.selectbox("Grup 1", grp_opts,
                                 index=grp_opts.index(d_g1), key="t1g1")
        with gc2:
            t1_g2 = st.selectbox("Grup 2", grp_opts,
                                 index=grp_opts.index(d_g2), key="t1g2")

        # Word'de görünecek isimler
        ll1, ll2 = st.columns(2)
        with ll1:
            t1_label1 = st.text_input("Grup 1 etiketi (manuscript)",
                                       value=str(t1_g1), key="t1l1")
        with ll2:
            t1_label2 = st.text_input("Grup 2 etiketi (manuscript)",
                                       value=str(t1_g2), key="t1l2")

        # Değişken seçimi
        st.markdown("**Değişken seçimi**")
        avail_num = [c for c in (["YAS"] + LAB_COLS + HEMA_COLS +
                                  INDEX_COLS) if c in df.columns]
        t1_num_vars = st.multiselect(
            "Sürekli değişkenler",
            options=avail_num,
            default=[v for v in ["YAS", "WBC", "NE#", "LY#", "MO#", "PLT",
                                 "HGB", "HCT", "MCV", "RDW-CV",
                                 "NLR", "PLR", "SII", "SIRI", "AISI"]
                     if v in avail_num],
            key="t1nv",
        )
        avail_cat = [c for c in ["CINSIYET_LBL", "YAS_GRUBU",
                                  "B12_KAT", "VITD_KAT"]
                     if c in df.columns and c != t1_group_col]
        t1_cat_vars = st.multiselect(
            "Kategorik değişkenler",
            options=avail_cat,
            default=[v for v in ["CINSIYET_LBL"] if v in avail_cat],
            key="t1cv",
        )

        if st.button("📊 Tablo 1'i Oluştur", type="primary", key="gen_t1"):
            if t1_g1 == t1_g2:
                st.warning("Lütfen iki farklı grup seçin.")
            elif not t1_num_vars and not t1_cat_vars:
                st.warning("En az bir değişken seçin.")
            else:
                tbl = build_table_one(
                    df, t1_group_col, t1_g1, t1_g2,
                    t1_num_vars, t1_cat_vars,
                )
                if tbl.empty:
                    st.error("Tablo oluşturulamadı — yeterli veri yok.")
                else:
                    # Etiketleri kullanıcı tercihine göre değiştir
                    rename_cols = {
                        f"{t1_g1} (n = {(df[t1_group_col]==t1_g1).sum()})":
                            f"{t1_label1} (n = {(df[t1_group_col]==t1_g1).sum()})",
                        f"{t1_g2} (n = {(df[t1_group_col]==t1_g2).sum()})":
                            f"{t1_label2} (n = {(df[t1_group_col]==t1_g2).sum()})",
                    }
                    tbl = tbl.rename(columns=rename_cols)

                    # Effect size etiketini gizle (görünmesin ama colspan için kalsın)
                    display_tbl = tbl.drop(columns=["_es_label"])

                    st.markdown(
                        f"#### Table 1 — Comparison of characteristics "
                        f"between {t1_label1} and {t1_label2} groups"
                    )
                    st.dataframe(display_tbl, use_container_width=True,
                                 hide_index=True)

                    # HTML preview (Word'e direkt yapıştırılabilir)
                    st.markdown("**🪟 HTML Önizleme (Word'e kopyala-yapıştır):**")
                    html_table = display_tbl.to_html(
                        index=False, escape=False,
                        classes="t1_table",
                        border=0,
                    )
                    # Profesyonel CSS
                    html_styled = f"""
                    <style>
                    .t1_wrap {{ font-family: 'Calibri', Arial, sans-serif;
                                font-size: 13px; color: #222; }}
                    .t1_table {{ border-collapse: collapse; width: 100%;
                                 margin-top: 8px; }}
                    .t1_table th {{ background:#f0f0f0; text-align:left;
                                    padding:8px 10px; font-weight:600;
                                    border-bottom:2px solid #444; }}
                    .t1_table td {{ padding:7px 10px;
                                    border-bottom:1px solid #ddd; }}
                    .t1_table tr:nth-child(even) {{ background:#fafafa; }}
                    .t1_title {{ font-weight:700; font-size:15px;
                                 margin-bottom:4px; }}
                    .t1_foot {{ font-size:11px; color:#444;
                                margin-top:10px; line-height:1.5; }}
                    </style>
                    <div class="t1_wrap">
                      <div class="t1_title">Table 1 — Comparison of characteristics
                          between {t1_label1} and {t1_label2} groups</div>
                      {html_table}
                      <div class="t1_foot">
                        <sup>1</sup> Mann–Whitney U test,
                        <sup>2</sup> Chi-square test with Yates' continuity
                        correction, <sup>3</sup> Independent Samples T-Test.
                        Categorical variables are presented as numbers and
                        percentages [n (%)], while continuous variables are
                        expressed as mean ± standard deviation or
                        median (1st quartile–3rd quartile), as appropriate.
                        Effect sizes were expressed as Cliff's Delta for
                        non-parametric continuous variables, Cohen's d for
                        parametric continuous variables, and Cramér's V for
                        categorical variables. p-value &lt; 0.05 was
                        considered statistically significant.
                      </div>
                    </div>
                    """
                    st.markdown(html_styled, unsafe_allow_html=True)

                    # ---- İndirme: HTML / Excel / CSV ----
                    st.markdown("**📥 İndirme Seçenekleri**")
                    dlt1, dlt2, dlt3 = st.columns(3)

                    # HTML (Word'e yapıştırılabilir)
                    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Table 1</title>{html_styled}</head><body></body></html>"""
                    with dlt1:
                        st.download_button(
                            "📄 HTML (Word'e aç)",
                            data=full_html,
                            file_name=f"Table1_{t1_label1}_vs_{t1_label2}.html",
                            mime="text/html", key="dl_t1_html",
                        )

                    # Excel
                    bxl = io.BytesIO()
                    with pd.ExcelWriter(bxl, engine="xlsxwriter") as w:
                        display_tbl.to_excel(
                            w, index=False, sheet_name="Table 1")
                        # Format
                        wb = w.book
                        ws = w.sheets["Table 1"]
                        hdr_fmt = wb.add_format({
                            "bold": True, "bg_color": "#E8E8E8",
                            "border": 1, "align": "left",
                            "valign": "vcenter",
                        })
                        cell_fmt = wb.add_format({
                            "border": 1, "valign": "vcenter",
                            "text_wrap": True,
                        })
                        for col_i, col_name in enumerate(display_tbl.columns):
                            ws.write(0, col_i, col_name, hdr_fmt)
                            max_w = max(
                                len(str(col_name)),
                                int(display_tbl[col_name].astype(str)
                                    .str.len().max() or 10),
                            ) + 2
                            ws.set_column(col_i, col_i,
                                          min(max_w, 38), cell_fmt)
                    with dlt2:
                        st.download_button(
                            "📊 Excel (.xlsx)",
                            data=bxl.getvalue(),
                            file_name=f"Table1_{t1_label1}_vs_{t1_label2}.xlsx",
                            mime=("application/vnd.openxmlformats-"
                                  "officedocument.spreadsheetml.sheet"),
                            key="dl_t1_xl",
                        )

                    # CSV
                    with dlt3:
                        st.download_button(
                            "📝 CSV (UTF-8)",
                            data=display_tbl.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"Table1_{t1_label1}_vs_{t1_label2}.csv",
                            mime="text/csv", key="dl_t1_csv",
                        )

                    st.success(
                        "✅ HTML dosyasını çift tıklayarak Word ile açabilirsiniz "
                        "— tablo formatlı olarak yapışacaktır."
                    )

# --- Sekme 11: İndir/Rapor --------------------------------------------------
with tabs[10]:
    st.subheader("📥 Sonuçları Excel Olarak İndir")

    # Tüm tabloları hazırla
    norm_tbl_full = normality_table(df, ALL_NUM)
    norm_map_full = dict(zip(norm_tbl_full["Değişken"], norm_tbl_full["Yorum"] == "Normal"))

    bundle = {
        "Veri_Islenmis": df,
        "Tanimlayici": descriptive(df, ALL_NUM),
        "Normallik": norm_tbl_full,
        "Frek_Cinsiyet":  freq_table(df["CINSIYET_LBL"], "Cinsiyet") if "CINSIYET_LBL" in df.columns else pd.DataFrame(),
        "Frek_YasGrubu":  freq_table(df["YAS_GRUBU"],   "Yaş Grubu"),
        "Frek_B12":       freq_table(df["B12_KAT"],     "B12"),
        "Frek_VitD":      freq_table(df["VITD_KAT"],    "VitD"),
        "Karsi_B12":      two_group_test(df, "B12_KAT", ALL_NUM, "Eksik","Normal", norm_map_full, correction),
        "Karsi_VitD":     two_group_test(df, "VITD_KAT",ALL_NUM, "Eksik","Yeterli",norm_map_full, correction),
        "Karsi_Cinsiyet": two_group_test(df, "CINSIYET_LBL", ALL_NUM, "Erkek","Kız", norm_map_full, correction),
        "Kruskal_YasGrubu":multi_group_test(df, "YAS_GRUBU", ALL_NUM, norm_map_full, correction),
        "Kor_B12_spearman":  corr_table(df, "B12",  ALL_NUM, "spearman"),
        "Kor_VitD_spearman": corr_table(df, "VITD", ALL_NUM, "spearman"),
    }
    excel_bytes = to_excel_download(bundle)
    st.download_button(
        "📊 Tüm sonuçları tek Excel dosyasında indir",
        data=excel_bytes,
        file_name="b12_vitd_pediatrik_analiz.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.success("Sonuçlar hazır.")

    # ---------------------------------------------------------------
    # ORANGE-UYUMLU CSV EXPORT (sınıflandırma için hazır tek dosya)
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("🍊 Orange için hazır CSV (sınıflandırma)")
    st.caption(
        "İndeksler + hedef sütunlar hazır. Ham B12/VitD değerleri leakage'ı "
        "önlemek için bilerek ÇIKARILDI. Dosyayı doğrudan Orange 'File' "
        "widget'ına verebilirsiniz."
    )

    # Öngörücüler: hemogram + indeksler (PIV=AISI olduğu için PIV çıkarıldı)
    orange_pred = [c for c in HEMA_COLS if c in df.columns] + \
                  [c for c in INDEX_COLS if c in df.columns and c != "PIV"]

    od = df[orange_pred].copy()

    # Hedef 1: B12 (Eksik / Normal)
    od["B12_HEDEF"] = df["B12_EKSIK"].map({1: "Eksik", 0: "Normal"})
    # Hedef 2: Vitamin D (Eksik / Yeterli)
    od["VITD_HEDEF"] = df["VITD_EKSIK"].map({1: "Eksik", 0: "Yeterli"})

    # Hedef 3: Kombine 4 grup
    def _kombine_kat(b, d):
        if pd.isna(b) or pd.isna(d):
            return np.nan
        b, d = int(b), int(d)
        if b == 0 and d == 0: return "Kontrol"
        if b == 0 and d == 1: return "Sadece D eksik"
        if b == 1 and d == 0: return "Sadece B12 eksik"
        return "Kombine eksik"
    od["KOMBINE_KAT"] = [
        _kombine_kat(b, d) for b, d in zip(df["B12_EKSIK"], df["VITD_EKSIK"])
    ]

    # Meta sütunlar (stratifikasyon/filtre için; ÖNGÖRÜCÜ DEĞİL)
    if "YAS" in df.columns:          od["YAS"] = df["YAS"].values
    if "YAS_GRUBU" in df.columns:    od["YAS_GRUBU"] = df["YAS_GRUBU"].values
    if "CINSIYET_LBL" in df.columns: od["CINSIYET"] = df["CINSIYET_LBL"].values

    orange_csv = od.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "🍊 Orange-uyumlu CSV indir",
        data=orange_csv,
        file_name="orange_pediatrik_b12_vitd.csv",
        mime="text/csv",
    )
    st.caption(
        f"Öngörücü sayısı: {len(orange_pred)}  |  "
        f"Hedefler: B12_HEDEF, VITD_HEDEF, KOMBINE_KAT  |  "
        f"Satır: {len(od):,}"
    )
