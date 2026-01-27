# app.py
# Streamlit ML app: Hemogram -> predict B12 and Vitamin D (regression)
# Run: streamlit run app.py

from __future__ import annotations

import re
import os
import csv
from io import StringIO, BytesIO
# İstatistik kütüphaneleri
from scipy.stats import kruskal, f_oneway, shapiro


# --- GRAFİK İÇİN GEREKLİ KÜTÜPHANELER (YENİ EKLENDİ) ---
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import streamlit as st




st.set_page_config(page_title="Hemogram -> B12 & Vit D Tahmini", layout="wide")


# -----------------------------
# HELPERS (YARDIMCI FONKSİYONLAR)
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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    for col in df.columns:
        if col == "CINSIYET": continue
        if col == "PROTOKOL_NO": df[col] = df[col].astype(str); continue
        if df[col].dtype == object or "VİTAMİN" in col or col in {"B12", "HASTA_YAS"}:
            df[col] = safe_to_numeric(df[col])
    if "CINSIYET" in df.columns:
        s = df["CINSIYET"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        mapping = {"1": "E", "2": "K", "ERKEK": "E", "KADIN": "K", "MALE": "E", "FEMALE": "K"}
        df["CINSIYET"] = s.map(mapping).fillna(s).astype(str).str.upper()
    return df

def calculate_derived_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ne = df.get("NE#", np.nan); ly = df.get("LY#", np.nan); mo = df.get("MO#", np.nan)
    plt = df.get("PLT", np.nan); rbc = df.get("RBC", np.nan); mcv = df.get("MCV", np.nan)
    
    if "NLR" not in df.columns and "NE#" in df.columns and "LY#" in df.columns: df["NLR"] = ne / ly
    if "PLR" not in df.columns and "PLT" in df.columns and "LY#" in df.columns: df["PLR"] = plt / ly
    if "LMR" not in df.columns and "LY#" in df.columns and "MO#" in df.columns: df["LMR"] = ly / mo
    if "SII" not in df.columns and "PLT" in df.columns and "NE#" in df.columns and "LY#" in df.columns: df["SII"] = (plt * ne) / ly
    if "SIRI" not in df.columns and "NE#" in df.columns and "MO#" in df.columns and "LY#" in df.columns: df["SIRI"] = (ne * mo) / ly
    if "AISI" not in df.columns and "NE#" in df.columns and "PLT" in df.columns and "MO#" in df.columns: df["AISI"] = (ne * plt * mo) / ly
    if "Mentzer" not in df.columns and "MCV" in df.columns and "RBC" in df.columns: df["Mentzer"] = mcv / rbc
    return df.replace([np.inf, -np.inf], np.nan)

def segment_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "HASTA_YAS" not in df.columns: return df
    conditions = [(df['HASTA_YAS'] >= 0) & (df['HASTA_YAS'] <= 5), (df['HASTA_YAS'] >= 6) & (df['HASTA_YAS'] <= 11), (df['HASTA_YAS'] >= 12) & (df['HASTA_YAS'] <= 17)]
    choices = ['Okul Öncesi (0-5)', 'Okul Çağı (6-11)', 'Adolesan (12-17)']
    df['Yas_Grubu'] = np.select(conditions, choices, default='Diğer')
    return df

def segment_clinical_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "B12" in df.columns:
        conditions_b12 = [(df['B12'] < 200), (df['B12'] >= 200) & (df['B12'] <= 400), (df['B12'] > 400)]
        choices_b12 = ['1. Düşük (<200)', '2. Sınırda (200-400)', '3. Yüksek (>400)']
        df['B12_Grubu'] = np.select(conditions_b12, choices_b12, default='Diğer')
    if "VİTAMİN D" in df.columns:
        conditions_vitd = [(df['VİTAMİN D'] < 20), (df['VİTAMİN D'] >= 20) & (df['VİTAMİN D'] <= 30), (df['VİTAMİN D'] > 30)]
        choices_vitd = ['1. Eksiklik (<20)', '2. Yetersizlik (20-30)', '3. Yeterli (>30)']
        df['VitD_Grubu'] = np.select(conditions_vitd, choices_vitd, default='Diğer')
    return df

def generate_stat_table_advanced(df: pd.DataFrame, groups_col: str, params: list, force_parametric: bool = False):
    results = []
    if groups_col not in df.columns: return pd.DataFrame()
    
    valid_groups = sorted([g for g in df[groups_col].unique() if pd.notna(g) and str(g) != 'Diğer'])
    if len(valid_groups) < 2: return pd.DataFrame()

    group_counts = df[groups_col].value_counts()
    col_names = {g: f"{g} (n={group_counts.get(g, 0)})" for g in valid_groups}
    
    df_stat = df[df[groups_col].isin(valid_groups)].copy()

    for p in params:
        if p not in df_stat.columns: continue
        clean_col = df_stat.dropna(subset=[p])
        groups_data = [clean_col[clean_col[groups_col] == g][p] for g in valid_groups]
        if any(len(g) < 3 for g in groups_data): continue
        
        is_normal = False
        if not force_parametric:
            try:
                p_values = [shapiro(g)[1] for g in groups_data]
                is_normal = all(p > 0.05 for p in p_values)
            except: is_normal = False 
        
        row = {"Parametre": p}
        if force_parametric or is_normal:
            for g, data in zip(valid_groups, groups_data): row[col_names[g]] = f"{data.mean():.2f} ± {data.std():.2f}"
            try: _, p_val = f_oneway(*groups_data); test_desc = "ANOVA"
            except: p_val = 1.0; test_desc = "Hata"
        else:
            for g, data in zip(valid_groups, groups_data): row[col_names[g]] = f"{data.median():.2f} ({data.min():.2f} - {data.max():.2f})"
            try: _, p_val = kruskal(*groups_data); test_desc = "Kruskal-Wallis"
            except: p_val = 1.0; test_desc = "Hata"

        row["P Değeri"] = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
        row["Metod"] = test_desc
        results.append(row)
        
    return pd.DataFrame(results)

# --- GRAFİK ÇİZME FONKSİYONU ---
def plot_group_comparison(df, group_col, value_col, force_parametric=False):
    """
    Nokta Dağılımı (Strip) + %95 Güven Aralığı (Point/CI) Grafiği.
    """
    valid_groups = sorted([g for g in df[group_col].unique() if pd.notna(g) and str(g) != 'Diğer'])
    plot_df = df[df[group_col].isin(valid_groups)].copy().dropna(subset=[value_col])
    if plot_df.empty: return None

    # Grafik Ayarları
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=len(valid_groups))
    
    # 1. Ham Veri Noktaları (Strip Plot)
    sns.stripplot(data=plot_df, x=group_col, y=value_col, order=valid_groups, jitter=0.2, alpha=0.5, size=3, palette=palette, ax=ax, zorder=0)

    # 2. Merkezi Eğilim ve %95 CI (Point Plot)
    estimator = np.mean if force_parametric else np.median
    est_label = "Ortalama" if force_parametric else "Medyan"
    
    # Siyah çizgiler (Güven Aralığı)
    sns.pointplot(data=plot_df, x=group_col, y=value_col, order=valid_groups, estimator=estimator, errorbar=('ci', 95),
                  color='black', capsize=0.1, join=False, markers="_", scale=0, err_kws={'linewidth': 2}, ax=ax, zorder=10)
    
    # Siyah Nokta (Merkez)
    sns.pointplot(data=plot_df, x=group_col, y=value_col, order=valid_groups, estimator=estimator, errorbar=None,
                  color='black', join=False, markers="D", scale=0.8, ax=ax, zorder=11)

    # 3. İstatistik (Başlık için)
    groups_data = [plot_df[plot_df[group_col] == g][value_col] for g in valid_groups]
    p_text = "N/A"
    try:
        if force_parametric: _, p_val = f_oneway(*groups_data); test_name = "ANOVA"
        else: _, p_val = kruskal(*groups_data); test_name = "Kruskal-Wallis"
        p_text = "< 0.001" if p_val < 0.001 else f"{p_val:.3f}"
    except: pass
    
    ax.set_title(f"{value_col} - {est_label} ve %95 CI\n{test_name} P: {p_text}", fontsize=14, fontweight='bold')
    ax.set_xlabel(group_col, fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    sns.despine()
    return fig

@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes: bytes, filename: str, encoding: str, user_sep: str):
    ext = os.path.splitext(filename.lower())[1]
    if ext in [".xlsx", ".xls"]:
        try:
            bio = BytesIO(file_bytes)
            return pd.read_excel(bio), "excel", None
        except Exception as e: raise ValueError(f"Excel hatası: {e}")
    text = file_bytes.decode(encoding, errors="replace")
    try: sep = csv.Sniffer().sniff(text[:20000], delimiters=[",", ";", "\t", "|"]).delimiter
    except: sep = user_sep if user_sep else ";"
    bad_lines = []
    return pd.read_csv(StringIO(text), sep=sep, engine="python", on_bad_lines=lambda l: bad_lines.append(l) or None), f"csv(sep='{sep}')", bad_lines

# --- KORELASYON HEATMAP FONKSİYONU ---
def plot_correlation_heatmap(df, cols):
    """
    Seçilen sütunlar için Spearman korelasyon matrisi çizer.
    Spearman seçilme nedeni: Biyolojik verilerde doğrusal olmayan ilişkileri daha iyi yakalar.
    """
    # Sadece sayısal sütunları al
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2: return None
    
    corr_df = df[valid_cols].dropna()
    if corr_df.empty: return None
    
    # Korelasyonu hesapla (Spearman)
    corr = corr_df.corr(method='spearman')
    
    # Grafik
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Üst üçgeni gizle (daha temiz görünüm)
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title("Spearman Korelasyon Matrisi", fontsize=14, fontweight='bold')
    return fig

# --- ROC ANALİZİ FONKSİYONU ---
def perform_roc_analysis(df, target_vitamin, threshold, feature_cols, condition_type='less'):
    """
    Belirli bir vitamin eksikliğini (Target) tahmin etmede 
    hemogram parametrelerinin başarısını (AUC) hesaplar.
    
    condition_type='less': Target < threshold ise 'Eksiklik Var (1)' kabul edilir.
    """
    results = []
    
    # Veri Hazırlığı
    temp_df = df.dropna(subset=[target_vitamin] + feature_cols).copy()
    if temp_df.empty: return None, None
    
    # Binary Hedef Oluştur (1: Hasta/Eksik, 0: Sağlam)
    if condition_type == 'less':
        y_true = (temp_df[target_vitamin] < threshold).astype(int)
    else:
        y_true = (temp_df[target_vitamin] > threshold).astype(int)
        
    # Sınıf dağılımı kontrolü (Sadece 0 veya Sadece 1 varsa ROC çizilemez)
    if len(np.unique(y_true)) < 2:
        return "Tek sınıf hatası", None

    # Grafik Hazırlığı
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Her bir özellik için tek tek ROC hesapla
    for feature in feature_cols:
        y_score = temp_df[feature]
        
        # ROC Hesapla
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Eğer AUC < 0.5 ise (Ters ilişki), skoru ters çevirip tekrar hesapla
        # Örn: B12 düştükçe MCV artar. MCV için AUC yüksek çıkmalı.
        # Ancak HGB düşebilir. Bu durumda AUC < 0.5 çıkar. 
        # Klinikte 'tanısal güç' önemlidir, yönü farketmeksizin AUC > 0.5 görmek isteriz.
        if roc_auc < 0.5:
            fpr, tpr, _ = roc_curve(y_true, -y_score) # Skoru ters çevir
            roc_auc = auc(fpr, tpr)
            label = f"{feature} (Inverse) (AUC = {roc_auc:.3f})"
        else:
            label = f"{feature} (AUC = {roc_auc:.3f})"
            
        results.append({"Parametre": feature, "AUC": roc_auc})
        
        # Sadece AUC değeri 0.55'ten büyük olanları çiz (Grafiği boğmamak için) veya en iyi 5'i
        if roc_auc > 0.55:
            ax.plot(fpr, tpr, label=label)

    # Referans çizgisi
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(f'{target_vitamin} Eksikliği (<{threshold}) İçin ROC Eğrileri')
    ax.legend(loc="lower right")
    
    # Sonuç Tablosu
    res_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    
    return res_df, fig

# -----------------------------
# UI (ARAYÜZ)
# -----------------------------
st.title("Hemogram -> B12 ve Vitamin D Tahmini (Regresyon)")

with st.sidebar:
    st.header("Veri")
    uploaded = st.file_uploader("Dosya yükle (XLSX / CSV)", type=["xlsx", "xls", "csv"])
    sep = st.selectbox("CSV ayırıcı", [",", ";", "\t", "|"], index=1)
    encoding = st.selectbox("Encoding", ["utf-8", "utf-8-sig", "cp1254", "latin1"], index=1)
    
    st.divider()
    st.header("Analiz ve İstatistik")
    target_choice = st.radio("Hedef", ["B12", "VİTAMİN D"], index=0)
    force_para = st.checkbox("Normallik Testini Yoksay (Hepsini Mean ± SD Ver)", value=False)
    do_multitarget = st.checkbox("Çoklu Hedef Raporu", value=True)


if uploaded is None:
    st.info("Lütfen sol menüden dosyanızı yükleyin."); st.stop()

file_bytes = uploaded.getvalue()
try: df_raw, read_mode, bad_lines = read_uploaded_file(file_bytes, uploaded.name, encoding, sep)
except Exception as e: st.error(f"Dosya okunamadı: {e}"); st.stop()

st.success(f"Dosya okundu ✅ ({len(df_raw)} satır)")
df = clean_dataframe(df_raw)
df = calculate_derived_indices(df)
if "HASTA_YAS" in df.columns: df = segment_age_groups(df)
df = segment_clinical_groups(df)

# --- İSTATİSTİK ve GRAFİK BÖLÜMÜ ---
st.divider()
st.header("📋 Detaylı Klinik İstatistikler ve Grafikler")

target_params = ["B12", "VİTAMİN D", "WBC", "HGB", "HCT", "MCV", "PLT", "NE#", "LY#", "MO#", "EO#", "BA#", "RDW-CV", "RDW-SD", "MPV", "PCT", "PDW", "NLR", "PLR", "LMR", "SII", "SIRI", "AISI", "Mentzer"]
present_params = [p for p in target_params if p in df.columns]

group_options = {}
if "Yas_Grubu" in df.columns: group_options["Yaş Grupları"] = "Yas_Grubu"
if "B12_Grubu" in df.columns: group_options["B12 Durumu"] = "B12_Grubu"
if "VitD_Grubu" in df.columns: group_options["Vitamin D Durumu"] = "VitD_Grubu"

if group_options:
    selected_label = st.radio("Analiz Kriteri:", list(group_options.keys()), horizontal=True)
    selected_group_col = group_options[selected_label]
    
    # 1. Tablo
    stat_table = generate_stat_table_advanced(df, selected_group_col, present_params, force_parametric=force_para)
    if not stat_table.empty:
        st.dataframe(stat_table, use_container_width=True, hide_index=True)
        st.download_button("Tabloyu İndir (CSV)", stat_table.to_csv(index=False, sep=";").encode('utf-8-sig'), f"istatistik_{selected_group_col}.csv", "text/csv")
        
        # 2. Grafik (YENİ KISIM)
        st.divider()
        st.subheader("📊 Grafiksel Analiz (%95 CI)")
        graph_param = st.selectbox("Grafiğini çizmek istediğiniz parametre:", options=present_params, index=0)
        
        if graph_param:
            with st.spinner("Grafik oluşturuluyor..."):
                fig = plot_group_comparison(df, selected_group_col, graph_param, force_parametric=force_para)
                if fig:
                    st.pyplot(fig)
# ... (Önceki grafik kodlarının bitişi) ...

# -----------------------------
# EK ANALİZLER: KORELASYON & ROC
# -----------------------------
st.divider()
st.header("🔍 İleri Analizler: Korelasyon ve ROC")

tab1, tab2 = st.tabs(["🔥 Korelasyon Heatmap", "🎯 ROC Analizi (Tanısal Güç)"])

# --- TAB 1: HEATMAP ---
with tab1:
    st.markdown("Seçilen parametreler arasındaki ilişkiyi (Spearman Korelasyonu) gösterir.")
    
    # Varsayılan olarak mantıklı parametreleri seçelim
    default_cols = ["B12", "VİTAMİN D", "HGB", "MCV", "WBC", "PLT", "NE#", "LY#", "NLR"]
    valid_defaults = [c for c in default_cols if c in df.columns]
    
    selected_corr_cols = st.multiselect(
        "Korelasyon haritasına dahil edilecek parametreleri seçin:",
        options=present_params,
        default=valid_defaults
    )
    
    if st.button("Heatmap Oluştur"):
        with st.spinner("Matris hesaplanıyor..."):
            fig_corr = plot_correlation_heatmap(df, selected_corr_cols)
            if fig_corr:
                st.pyplot(fig_corr)
            else:
                st.warning("Yeterli veri seçilmedi.")

# --- TAB 2: ROC ANALİZİ ---
with tab2:
    st.markdown("""
    Bu modül, hemogram parametrelerinin **Vitamin Eksikliğini** tespit etme başarısını (AUC) ölçer.
    * **AUC yakşalık 1.0:** Mükemmel ayırıcı.
    * **AUC yaklaşık 0.5:** Yazı-tura (Ayırt ediciliği yok).
    """)
    
    c1, c2 = st.columns(2)
    roc_target = c1.selectbox("Hangi eksiklik analiz edilecek?", ["B12 Eksikliği", "D Vitamini Eksikliği"])
    
    # Eşik Değerler (Kullanıcı değiştirebilsin)
    if roc_target == "B12 Eksikliği":
        target_col = "B12"
        threshold = c2.number_input("B12 Eksiklik Sınırı (pg/mL)", value=200)
    else:
        target_col = "VİTAMİN D"
        threshold = c2.number_input("Vit D Eksiklik Sınırı (ng/mL)", value=20)
        
    roc_features = st.multiselect(
        "Tanısal gücü test edilecek parametreler:",
        options=[p for p in present_params if p not in ["B12", "VİTAMİN D"]], # Hedefin kendisini çıkar
        default=["MCV", "HGB", "WBC", "NE#", "LY#", "NLR", "PLR"] # Örnek varsayılanlar
    )
    
    if st.button("ROC Analizini Çalıştır"):
        if target_col in df.columns and roc_features:
            res_df, fig_roc = perform_roc_analysis(df, target_col, threshold, roc_features)
            
            if isinstance(res_df, str): # Hata mesajı döndüyse
                st.error(f"Hata: {res_df} (Veri setinde sadece 'Hasta' veya sadece 'Sağlam' kişiler olabilir).")
            elif res_df is not None:
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.write("**AUC Skor Tablosu**")
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                with col_right:
                    st.pyplot(fig_roc)
                    st.caption("Not: Ters ilişki gösteren parametreler (örn. B12 düşerken artanlar) otomatik olarak düzeltilmiştir.")
            else:
                st.warning("Veri yetersiz.")
        else:
            st.error("Seçilen hedef sütun veride bulunamadı.")
                    st.caption(f"**Grafik Notu:** {graph_param} parametresinin {selected_label} gruplarına göre dağılımı. Siyah çizgiler %95 Güven Aralığını gösterir.")
                else: st.warning("Grafik için yeterli veri yok.")
    else: st.warning("Yeterli veri yok.")
else: st.warning("Gruplama verisi bulunamadı.")

