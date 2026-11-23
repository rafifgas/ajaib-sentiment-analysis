import streamlit as st
import joblib
from pathlib import Path

# ---------- PAGE ----------
st.set_page_config(
    page_title="AjaibSentiment | Analisis Ulasan",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "model_sentimen_ajaib.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ---------- STYLE ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 8% 0%, rgba(75, 116, 255, .13), transparent 28%),
      radial-gradient(circle at 95% 8%, rgba(25, 190, 160, .10), transparent 25%),
      #f5f7fc;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }
[data-testid="stSidebar"] { background: #101a35; }
[data-testid="stSidebar"] * { color: #edf2ff !important; }
[data-testid="stSidebar"] .stMarkdown { color: #c6d2ef; }
.hero {
    background: linear-gradient(115deg, #111d3c 0%, #203d7b 58%, #236b86 100%);
    padding: 30px 34px; border-radius: 24px; color: white;
    box-shadow: 0 18px 45px rgba(25, 49, 103, .18);
    margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero:after {
    content: ""; position: absolute; width: 260px; height: 260px;
    right: -65px; top: -115px; border: 1px solid rgba(255,255,255,.18);
    border-radius: 50%; box-shadow: 0 0 0 28px rgba(255,255,255,.04), 0 0 0 58px rgba(255,255,255,.03);
}
.eyebrow { color: #9fd8ff; font-size: 12px; letter-spacing: 2px; font-weight: 800; text-transform: uppercase; }
.hero h1 { color: white; font-family: 'Manrope', sans-serif; font-size: clamp(27px, 4vw, 42px); font-weight: 800; margin: 8px 0 8px; }
.hero p { color: #d6e4ff; font-size: 15px; max-width: 720px; margin: 0; }
.pill {
    display: inline-block; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
    color: #eaf4ff; padding: 7px 12px; border-radius: 99px; font-size: 12px; margin-top: 18px;
}
.section-title { color: #152344; font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 20px; margin: 8px 0 4px; }
.muted { color: #71809d; font-size: 13px; }
.panel {
    background: white; border: 1px solid #e7ebf4; border-radius: 18px;
    padding: 22px 24px; box-shadow: 0 8px 25px rgba(25, 45, 85, .045);
}
.metric {
    background: white; border: 1px solid #e7ebf4; border-radius: 17px;
    padding: 17px 19px; box-shadow: 0 7px 22px rgba(25,45,85,.04); min-height: 104px;
}
.metric-label { color: #71809d; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; }
.metric-value { color: #17294f; font-family: 'Manrope', sans-serif; font-size: 27px; font-weight: 800; margin-top: 5px; }
div.stButton > button {
    background: linear-gradient(100deg, #315be8, #247caa); color: white; border: 0;
    border-radius: 12px; min-height: 48px; font-weight: 800; box-shadow: 0 8px 18px rgba(49,91,232,.2);
}
div.stButton > button:hover { color: white; border: 0; filter: brightness(1.06); }
.stTextArea textarea {
    border: 1px solid #dce4f2; border-radius: 13px; background: #fbfcff;
    font-size: 15px; padding: 14px;
}
.result {
    border-radius: 16px; padding: 20px 22px; margin: 12px 0 18px;
    border: 1px solid #dce6f7; background: #f7faff;
}
.result-positive { background: #effaf5; border-color: #c9eedc; }
.result-negative { background: #fff3f2; border-color: #f6d5d1; }
.result-neutral { background: #fff9eb; border-color: #f3e4b9; }
.result-label { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; font-weight: 800; color: #71809d; }
.result-value { font-family: 'Manrope', sans-serif; font-size: 30px; font-weight: 800; margin-top: 3px; }
.footer { color: #8490a7; font-size: 12px; text-align: center; padding: 25px 0 5px; }
div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #315be8, #22a6a1); }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 📈 AjaibSentiment")
    st.caption("Sistem klasifikasi opini pengguna")
    st.markdown("---")
    st.markdown("### Tentang sistem")
    st.write("Aplikasi ini memprediksi sentimen komentar pengguna aplikasi Ajaib ke dalam tiga kategori.")
    st.markdown("**Metode yang digunakan**")
    st.write("• TF-IDF untuk ekstraksi fitur teks")
    st.write("• Multinomial Naive Bayes untuk klasifikasi")
    st.markdown("---")
    st.markdown("### Kategori sentimen")
    st.write("🟢 **Positif** — komentar bernada baik atau puas")
    st.write("🔴 **Negatif** — komentar berisi keluhan atau kritik")
    st.write("🟡 **Netral** — komentar informatif atau tidak menunjukkan kecenderungan kuat")
    st.markdown("---")
    st.caption("Proyek analisis sentimen ulasan aplikasi Ajaib")

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Natural Language Processing · Machine Learning</div>
  <h1>Memahami Suara Pengguna Ajaib</h1>
  <p>Analisis komentar secara otomatis untuk mengidentifikasi opini pengguna.
  Masukkan sebuah ulasan dan lihat hasil klasifikasi sentimennya dalam beberapa detik.</p>
  <div class="pill">✦ TF-IDF &nbsp; + &nbsp; Multinomial Naive Bayes</div>
</div>
""", unsafe_allow_html=True)

# ---------- MODEL CHECK ----------
if not MODEL_PATH.exists():
    st.error("File model belum ditemukan. Letakkan `model_sentimen_ajaib.pkl` di folder yang sama dengan `app.py`.")
    st.stop()

model = load_model()

# ---------- METRICS ----------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="metric"><div class="metric-label">Algoritma</div><div class="metric-value">Naive Bayes</div><div class="muted">Multinomial classifier</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric"><div class="metric-label">Ekstraksi fitur</div><div class="metric-value">TF-IDF</div><div class="muted">Unigram & bigram</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric"><div class="metric-label">Akurasi pengujian</div><div class="metric-value">76%</div><div class="muted">Pada data uji yang digunakan</div></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.markdown('<div class="section-title">🔎 Uji sentimen komentar</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Ketik komentar secara manual atau pilih contoh untuk mencoba sistem.</div>', unsafe_allow_html=True)
    sample_choice = st.selectbox(
        "Contoh komentar (opsional)",
        ["Tulis komentar sendiri", "Aplikasi mudah digunakan dan sangat membantu", "Penarikan dana saya lama sekali, mohon diperbaiki", "Aplikasi ini biasa saja, belum ada komentar lain"],
        label_visibility="collapsed"
    )
    sample_map = {
        "Tulis komentar sendiri": "",
        "Aplikasi mudah digunakan dan sangat membantu": "Aplikasi mudah digunakan dan sangat membantu",
        "Penarikan dana saya lama sekali, mohon diperbaiki": "Penarikan dana saya lama sekali, mohon diperbaiki",
        "Aplikasi ini biasa saja, belum ada komentar lain": "Aplikasi ini biasa saja, belum ada komentar lain"
    }
    default_text = sample_map[sample_choice]
    komentar = st.text_area(
        "Komentar pengguna",
        value=default_text,
        placeholder="Contoh: Aplikasi Ajaib mudah dipakai, tetapi proses penarikan dana cukup lama...",
        height=155,
        label_visibility="collapsed"
    )
    analyze = st.button("✦  ANALISIS SENTIMEN", use_container_width=True)

with right:
    st.markdown('<div class="section-title">📌 Alur analisis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel">
      <p><b>01 · Input teks</b><br><span class="muted">Komentar dimasukkan ke sistem.</span></p>
      <p><b>02 · Ekstraksi fitur</b><br><span class="muted">TF-IDF mengubah teks menjadi representasi numerik.</span></p>
      <p><b>03 · Klasifikasi</b><br><span class="muted">Naive Bayes menentukan kategori sentimen.</span></p>
      <p style="margin-bottom:0"><b>04 · Hasil prediksi</b><br><span class="muted">Sistem menampilkan label dan skor probabilitas.</span></p>
    </div>
    """, unsafe_allow_html=True)

# ---------- PREDICTION ----------
if analyze:
    teks = komentar.strip()
    if not teks:
        st.warning("Masukkan komentar terlebih dahulu sebelum melakukan analisis.")
    else:
        prediksi = model.predict([teks])[0]
        probabilitas = model.predict_proba([teks])[0]
        kelas = model.classes_
        skor = dict(zip(kelas, probabilitas))

        st.write("")
        st.markdown('<div class="section-title">Hasil analisis komentar</div>', unsafe_allow_html=True)
        if prediksi.lower() == "positif":
            css_class, emoji = "result-positive", "🟢"
        elif prediksi.lower() == "negatif":
            css_class, emoji = "result-negative", "🔴"
        else:
            css_class, emoji = "result-neutral", "🟡"

        st.markdown(
            f'<div class="result {css_class}"><div class="result-label">Prediksi sentimen</div>'
            f'<div class="result-value">{emoji} {prediksi}</div></div>',
            unsafe_allow_html=True
        )

        p1, p2 = st.columns([1, 1])
        with p1:
            st.markdown("**Komentar yang dianalisis**")
            st.write(f"“{teks}”")
        with p2:
            st.markdown("**Skor probabilitas per kelas**")
            for nama_kelas, nilai in sorted(skor.items(), key=lambda x: x[1], reverse=True):
                st.write(f"{nama_kelas} · {nilai:.1%}")
                st.progress(float(nilai))

        st.caption("Skor probabilitas menunjukkan keluaran model, bukan jaminan bahwa prediksi selalu benar.")

st.markdown("""
<div class="footer">
  AjaibSentiment · Proyek Analisis Sentimen · Hasil prediksi dapat keliru dan perlu dievaluasi dengan data uji.
</div>
""", unsafe_allow_html=True)
