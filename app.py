import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="PANDU",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a6c 0%, #2563eb 100%);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 14px; color: white; text-align: center; margin-bottom: 2rem;
    }
    .main-header h1{font-size: 2rem; margin: 0 0 0.3rem 0;}
    .main-header p{font-size: 0.95rem; opacity: 0.85; margin: 0;}

    .result-graduate {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        padding: 2rem; border-radius: 14px; color: white; text-align: center; margin-top: 1.5rem;
    }
    .result-dropout {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        padding: 2rem; border-radius: 14px; color: white; text-align: center; margin-top: 1.5rem;
    }
    .result-title {font-size: 2rem; font-weight: 800; margin: 0;}
    .result-prob {font-size: 3rem; font-weight: 900; margin: 0.4rem 0 0 0;}
    .result-subtitle {font-size: 0.95rem; opacity: 0.85; margin-top: 0.3rem;}

    .section-label {
        font-size: 0.75rem; font-weight: 700; letter-spacing: 1.2px;
        color: #6b7280; text-transform: uppercase;
        margin: 1.6rem 0 0.6rem 0;
        padding-bottom: 0.3rem; border-bottom: 1px solid #e5e7eb;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        color: white; font-weight: 700; font-size: 1.05rem;
        border: none; border-radius: 8px; padding: 0.7rem 2rem; width: 100%; margin-top: 1rem;
    }
    .stButton>button:hover { background: #1e40af;}

    .history-card {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;
    }
    .history-card .hc-top {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.4rem;
    }
    .history-card .hc-name { font-weight: 700; font-size: 0.95rem; color: #1e293b; }
    .history-card .hc-time { font-size: 0.78rem; color: #94a3b8; }
    .history-card .hc-badge-graduate {
        display: inline-block; background: #d1fae5; color: #065f46;
        border-radius: 20px; padding: 0.15rem 0.7rem; font-weight: 700; font-size: 0.82rem;
    }
    .history-card .hc-badge-dropout {
        display: inline-block; background: #fee2e2; color: #991b1b;
        border-radius: 20px; padding: 0.15rem 0.7rem; font-weight: 700; font-size: 0.82rem;
    }
    .history-card .hc-detail { font-size: 0.82rem; color: #64748b; margin-top: 0.3rem; }
    .compare-box {
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-radius: 10px; padding: 1rem 1.2rem; margin-top: 1rem;
    }
    .compare-box h4 { margin: 0 0 0.6rem 0; color: #0c4a6e; font-size: 0.9rem; }
    .compare-row { display: flex; justify-content: space-between; margin-bottom: 0.3rem; }
    .compare-model { font-weight: 700; font-size: 0.85rem; color: #1e293b; }
    .compare-val { font-size: 0.85rem; color: #475569; }
</style>
""", unsafe_allow_html=True)

MODEL_OPTIONS = {
    "Random Forest": "random_forest_pipeline.joblib",
    "Decision Tree": "decision_tree_pipeline.joblib",
    "Logistic Regression": "logistic_regression_tuned.joblib",
}

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    mdl = joblib.load(os.path.join(base, "random_forest.joblib"))
    meta = joblib.load(os.path.join(base, "model_metadata.joblib"))
    pre = mdl.named_steps['preprocessor']
    ohe = pre.named_transformers_['cat']
    valid = {
        'nationality': [int(x) for x in ohe.categories_[3]],
        'course': [int(x) for x in ohe.categories_[2]],
    }
    return mdl, meta.get("threshold", 0.3), valid

@st.cache_resource
def load_extra_model(filename):
    base = os.path.dirname(__file__)
    path = os.path.join(base, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

model, THRESHOLD, VALID = load_model()

NATIONALITY = dict(sorted({
    1: "Portugal",
    2: "Jerman",
    6: "Spanyol",
    11: "Italia",
    13: "Belanda",
    14: "Inggris",
    17: "Lithuania",
    21: "Angola",
    22: "Cape Verde",
    24: "Guinea",
    25: "Mozambik",
    26: "Sao Tome & Principe",
    41: "Turki",
    62: "Brazil",
    100: "Rumania",
    101: "Moldova",
    103: "Meksiko",
    105: "Ukraina",
    109: "Rusia",
}.items()))

NATIONALITY_DISPLAY = dict(sorted({
    "Indonesia": 62,
    "Malaysia": 62,
    "Singapura": 62,
    "Thailand": 62,
    "Vietnam": 62,
    "Filipina": 62,
    "India": 62,
    "Pakistan": 62,
    "Bangladesh": 62,
    "Sri Lanka": 62,
    "Nepal": 62,
    "Jepang": 62,
    "Korea Selatan": 62,
    "China": 62,
    "Taiwan": 62,
    "Arab Saudi": 62,
    "Iran": 62,
    "Timor Leste": 62,
    "Myanmar": 62,
    "Kamboja": 62,
    "Portugal": 1,
    "Jerman": 2,
    "Spanyol": 6,
    "Italia": 11,
    "Belanda": 13,
    "Inggris": 14,
    "Lithuania": 17,
    "Rumania": 100,
    "Moldova": 101,
    "Ukraina": 105,
    "Rusia": 109,
    "Brazil": 62,
    "Meksiko": 103,
    "Angola": 21,
    "Cape Verde": 22,
    "Guinea": 24,
    "Mozambik": 25,
    "Sao Tome": 26,
    "Turki": 41,
}.items()))

COURSE_DISPLAY = dict(sorted({
    "Teknik Informatika": 9119,
    "Animasi & Desain Multimedia": 171,
    "Desain Komunikasi Visual": 9070,
    "Manajemen Informatika (sore)": 9991,
    "Manajemen": 9147,
    "Pemasaran & Periklanan": 9670,
    "Pekerjaan Sosial": 9238,
    "Pekerjaan Sosial (sore)": 8014,
    "Keperawatan": 9500,
    "Kesehatan Gigi": 9556,
    "Keperawatan Hewan": 9085,
    "Agronomi": 9003,
    "Teknologi Biofuel": 33,
    "Equinkultura": 9130,
    "Jurnalisme & Komunikasi": 9773,
    "Pendidikan Dasar": 9853,
    "Pariwisata": 9254,
}.items()))

st.markdown("""
<div class="main-header">
  <h1>PANDU</h1>
  <p>Prediksi Akademik & Navigasi Dini Universitas</p>
</div>
""", unsafe_allow_html=True)

tab_predict, tab_history = st.tabs(["Prediksi", "Riwayat Prediksi"])

with tab_predict:

    st.markdown('<div class="section-label">Pilih Model</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        selected_model_name = st.selectbox(
            "Model yang digunakan",
            list(MODEL_OPTIONS.keys()),
            index=0,
            help="Pilih salah satu model untuk menjalankan prediksi."
        )
        selected_models = [selected_model_name]
    with col_m2:
        nama_mahasiswa = st.text_input("Nama Mahasiswa", placeholder="Nama...")

    st.markdown('<div class="section-label">Data Pribadi</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia saat Mendaftar", min_value=17, max_value=70, value=None, placeholder="Masukkan usia...")
        gender = st.selectbox("Jenis Kelamin", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    with col2:
        nat_label = st.selectbox("Kewarganegaraan", list(NATIONALITY_DISPLAY.keys()), index=None, placeholder="Pilih kewarganegaraan...")
        nationality_code = NATIONALITY_DISPLAY[nat_label] if nat_label is not None else None
        international = 0 if nationality_code == 1 else 1
     
        course_label = st.selectbox("Program Studi", list(COURSE_DISPLAY.keys()), index=None, placeholder="Pilih program studi...")
        course_code  = COURSE_DISPLAY[course_label] if course_label is not None else None
     

    st.markdown('<div class="section-label">Kondisi Ekonomi</div>', unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        tuition_fees = st.selectbox("Uang Kuliah Lancar?", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with col4:
        scholarship = st.selectbox("Penerima Beasiswa?", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Ya" if x == 1 else "Tidak")
    with col5:
        debtor = st.selectbox("Memiliki Tunggakan?", [1, 0], index=None, placeholder="Pilih...", format_func=lambda x: "Ya" if x == 1 else "Tidak")

    st.markdown('<div class="section-label">Kinerja Akademik - Semester 1</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cu1_enrolled = st.number_input("SKS (Sem 1)", min_value=0, max_value=30, value=None, placeholder="Masukkan angka...")
    with c2:
        cu1_approved = st.number_input("SKS Lulus (Sem 1)", min_value=0, max_value=30,value=None, placeholder="Masukkan angka...")
    with c3:
        cu1_ipk = st.number_input("IPK (Sem 1)", min_value=0.0, max_value=4.0, value=None, placeholder="Masukkan angka...", step=0.01, help="Skala 0.00 – 4.00")

    st.markdown('<div class="section-label">Kinerja Akademik – Semester 2</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        cu2_enrolled = st.number_input("SKS (Sem 2)", min_value=0, max_value=30, value=None, placeholder="Masukkan angka...")
    with c5:
        cu2_approved = st.number_input("SKS Lulus (Sem 2)", min_value=0, max_value=30, value=None, placeholder="Masukkan angka...")
    with c6:
        cu2_ipk = st.number_input("IPK (Sem 2)", min_value=0.0, max_value=4.0, value=None, placeholder="Masukkan angka...", step=0.01, help="Skala 0.00 – 4.00")

    predict_btn = st.button("Prediksi Kelulusan")

    if predict_btn:
        if not selected_models:
            st.warning("Pilih minimal satu model terlebih dahulu.")
            st.stop()

        input_data = pd.DataFrame([{
            "Age at enrollment": age,
            "Gender": gender,
            "Nacionality": nationality_code,
            "International": international,
            "Course": course_code,
            "Tuition fees up to date": tuition_fees,
            "Scholarship holder": scholarship,
            "Debtor": debtor,
            "Admission grade": 130.0,
            "Curricular units 1st sem (enrolled)": cu1_enrolled,
            "Curricular units 1st sem (approved)": cu1_approved,
            "Curricular units 1st sem (grade)": cu1_ipk * 5.0,
            "Curricular units 1st sem (evaluations)": max(cu1_enrolled, cu1_approved),
            "Curricular units 2nd sem (enrolled)": cu2_enrolled,
            "Curricular units 2nd sem (approved)": cu2_approved,
            "Curricular units 2nd sem (grade)": cu2_ipk * 5.0,
            "Curricular units 2nd sem (evaluations)": max(cu2_enrolled, cu2_approved),
            "Marital status": 1,
            "Application mode": 1,
            "Application order": 1,
            "Daytime/evening attendance": 1,
            "Previous qualification": 1,
            "Previous qualification (grade)": 130.0,
            "Mother's qualification": 1,
            "Father's qualification": 1,
            "Mother's occupation": 4,
            "Father's occupation": 4,
            "Displaced": 0,
            "Educational special needs": 0,
            "Curricular units 1st sem (credited)": 0,
            "Curricular units 1st sem (without evaluations)": 0,
            "Curricular units 2nd sem (credited)": 0,
            "Curricular units 2nd sem (without evaluations)": 0,
            "Unemployment rate": 10.8,
            "Inflation rate": 1.4,
            "GDP": 1.74,
        }])

        hasil_per_model = {}
        base = os.path.dirname(__file__)

        for model_name in selected_models:
            filename = MODEL_OPTIONS[model_name]
            mdl_path = os.path.join(base, filename)

            if model_name == "Random Forest":
                mdl_used = model
            else:
                mdl_used = load_extra_model(filename)

            if mdl_used is None:
                st.error(f"File model '{filename}' tidak ditemukan.")
                continue

            try:
                expected_cols = list(mdl_used.feature_names_in_)
                input_aligned = input_data[expected_cols]
                proba = mdl_used.predict_proba(input_aligned)[0]
                prob_graduate = proba[0]
                prob_dropout = proba[1]
                prediction = "graduate" if prob_dropout < THRESHOLD else "dropout"
                hasil_per_model[model_name] = {
                    "prob_graduate": prob_graduate,
                    "prob_dropout": prob_dropout,
                    "prediction": prediction,
                }
            except Exception as e:
                st.error(f"Error pada model {model_name}: {e}")
                continue

        if not hasil_per_model:
            st.stop()

        if len(hasil_per_model) == 1:
            model_name, res = list(hasil_per_model.items())[0]
            if res["prediction"] == "graduate":
                st.markdown(f"""
                <div class="result-graduate">
                  <div class="result-title">DIPREDIKSI LULUS</div>
                  <div class="result-prob">{res['prob_graduate']:.0%}</div>
                  <div class="result-subtitle">Probabilitas mahasiswa ini menyelesaikan studi &nbsp;|&nbsp; Model: {model_name}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-dropout">
                  <div class="result-title">RISIKO DROPOUT</div>
                  <div class="result-prob">{res['prob_dropout']:.0%}</div>
                  <div class="result-subtitle">Probabilitas mahasiswa ini tidak menyelesaikan studi &nbsp;|&nbsp; Model: {model_name}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-label">Perbandingan Hasil Antar Model</div>', unsafe_allow_html=True)
            compare_html = '<div class="compare-box"><h4>Hasil Prediksi Semua Model</h4>'
            for mname, res in hasil_per_model.items():
                badge_cls = "hc-badge-graduate" if res["prediction"] == "graduate" else "hc-badge-dropout"
                badge_txt = "LULUS" if res["prediction"] == "graduate" else "DROPOUT"
                prob_val  = f"{res['prob_graduate']:.1%}" if res["prediction"] == "graduate" else f"{res['prob_dropout']:.1%}"
                compare_html += f"""
                <div class="compare-row">
                  <span class="compare-model">{mname}</span>
                  <span class="compare-val">
                    <span class="{badge_cls}">{badge_txt}</span>&nbsp; Probabilitas: {prob_val}
                  </span>
                </div>"""
            compare_html += "</div>"
            st.markdown(compare_html, unsafe_allow_html=True)

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        label = nama_mahasiswa.strip() if nama_mahasiswa and nama_mahasiswa.strip() else f"Mahasiswa #{len(st.session_state.prediction_history) + 1}"
        record = {
            "label": label,
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            "prodi": course_label or "-",
            "nationality": nat_label or "-",
            "models": hasil_per_model,
        }
        st.session_state.prediction_history.insert(0, record)

        st.success("Hasil prediksi telah disimpan ke Riwayat Prediksi.")

with tab_history:
    st.markdown('<div class="section-label">Riwayat Prediksi</div>', unsafe_allow_html=True)

    if "prediction_history" not in st.session_state or not st.session_state.prediction_history:
        st.info("Belum ada riwayat prediksi. Jalankan prediksi terlebih dahulu.")
    else:
        col_info, col_clear = st.columns([4, 1])
        with col_info:
            st.caption(f"Total prediksi tersimpan: {len(st.session_state.prediction_history)}")
        with col_clear:
            if st.button("Hapus Semua"):
                st.session_state.prediction_history = []
                st.rerun()

        for i, rec in enumerate(st.session_state.prediction_history):
            model_summary_parts = []
            for mname, res in rec["models"].items():
                badge = "LULUS" if res["prediction"] == "graduate" else "DROPOUT"
                prob  = f"{res['prob_graduate']:.1%}" if res["prediction"] == "graduate" else f"{res['prob_dropout']:.1%}"
                model_summary_parts.append(f"{mname}: {badge} ({prob})")

            badges_html = ""
            for mname, res in rec["models"].items():
                if res["prediction"] == "graduate":
                    badges_html += f'<span class="hc-badge-graduate">LULUS</span> '
                else:
                    badges_html += f'<span class="hc-badge-dropout">DROPOUT</span> '

            detail_text = " &nbsp;|&nbsp; ".join(model_summary_parts)

            st.markdown(f"""
            <div class="history-card">
              <div class="hc-top">
                <span class="hc-name">{rec['label']}</span>
                <span class="hc-time">{rec['timestamp']}</span>
              </div>
              <div>{badges_html}</div>
              <div class="hc-detail">
                Prodi: {rec['prodi']} &nbsp;|&nbsp; Kewarganegaraan: {rec['nationality']}<br>
                {detail_text}
              </div>
            </div>
            """, unsafe_allow_html=True)
