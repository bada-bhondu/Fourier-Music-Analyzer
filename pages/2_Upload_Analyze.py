import streamlit as st
import numpy as np
import pandas as pd
from core.audio_loader import load_audio
from core.fourier_engine import (compute_fft_cached, compute_stft_cached,
                                  get_dominant_frequencies)
from utils.plot_utils import plot_waveform, plot_spectrum, plot_spectrogram
from utils.styles import inject_styles, section_badge, PLOTLY_LAYOUT

inject_styles()

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.2rem">
    <span style="font-size:2rem">🎵</span>
    <h1 style="margin:0">Upload & Analyze</h1>
</div>
<p style="color:#7070a0;font-size:0.95rem;margin-top:0.1rem;margin-bottom:1.5rem;
font-family:'Space Grotesk',sans-serif">
Load any audio file and inspect its time-domain and frequency-domain representations.
</p>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
if "audio" not in st.session_state:
    st.session_state.audio = None

# ── Upload ─────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop an audio file here, or click to browse",
    type=["wav", "mp3", "aac", "flac", "ogg", "m4a"],
    label_visibility="collapsed",
    help="Converted internally to mono float32 at the original sample rate."
)
st.markdown("""
<p style="color:#5a5a80;font-size:0.78rem;font-style:italic;
font-family:'Space Grotesk',sans-serif;margin-top:0.3rem">
Supported: WAV · MP3 · AAC · FLAC · OGG · M4A
</p>""", unsafe_allow_html=True)

if uploaded:
    with st.spinner("Loading audio..."):
        audio = load_audio(uploaded, uploaded.name)
        st.session_state.audio = audio
    st.markdown(f"""
    <div style="background:rgba(6,214,160,0.08);border:1px solid rgba(6,214,160,0.3);
    border-radius:12px;padding:0.75rem 1.2rem;margin-top:0.5rem;
    display:flex;align-items:center;gap:0.75rem">
        <span style="font-size:1.2rem">✅</span>
        <span style="color:#06d6a0;font-family:'Space Grotesk',sans-serif;
        font-weight:600;font-size:0.9rem">Loaded: {audio.file_name}</span>
    </div>""", unsafe_allow_html=True)

audio = st.session_state.audio
if audio is None:
    st.markdown("""
    <div style="background:#12121e;border:1px solid rgba(124,58,237,0.15);
    border-radius:14px;padding:2.5rem;text-align:center;margin-top:1rem">
        <p style="font-size:2.5rem;margin:0 0 0.5rem">🎶</p>
        <p style="color:#6060a0;font-family:'Space Grotesk',sans-serif;
        font-size:0.95rem;margin:0">
        Upload an audio file above to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Metadata panel ─────────────────────────────────────────────
st.divider()
section_badge("File Information")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sample Rate",  f"{audio.sample_rate:,} Hz")
c2.metric("Duration",     f"{audio.duration:.2f} s")
c3.metric("Samples",      f"{len(audio.signal):,}")
c4.metric("Channels",     f"{audio.num_channels} → mono")
c5.metric("Bit Depth",    str(audio.bit_depth) if audio.bit_depth else "N/A")

st.divider()

# ── Waveform ───────────────────────────────────────────────────
section_badge("Time Domain")
st.markdown("""<p style="color:#7070a0;font-size:0.85rem;
font-family:'Space Grotesk',sans-serif;margin-bottom:0.5rem">
Amplitude vs time — the raw digital signal as stored.
</p>""", unsafe_allow_html=True)

fig_wave = plot_waveform(audio)
fig_wave.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,18,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#c4c4e0"),
    height=220, margin=dict(t=30,b=35,l=55,r=20),
)
st.plotly_chart(fig_wave, use_container_width=True)
st.caption("Downsampled to 50,000 points for rendering. All analysis runs on the full signal.")

st.divider()

# ── FFT Spectrum ───────────────────────────────────────────────
section_badge("Frequency Domain — FFT")
st.markdown("""<p style="color:#7070a0;font-size:0.85rem;
font-family:'Space Grotesk',sans-serif;margin-bottom:0.75rem">
How much energy is present at each frequency — averaged over the entire file.
</p>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    window    = st.selectbox("Window", ["hann", "hamming", "blackman", "none"],
                              help="Hann is the standard for audio.")
with col2:
    log_scale = st.checkbox("dB scale", value=True,
                             help="Log scale reveals quiet components.")
with col3:
    max_hz = st.slider("Max frequency (Hz)", 1000, 22050, 20000, step=500)

with st.spinner("Computing FFT..."):
    fft_result = compute_fft_cached(audio.signal, audio.sample_rate, window)

fig_spec = plot_spectrum(fft_result, log_scale=log_scale, max_hz=max_hz)
fig_spec.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,18,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#c4c4e0"),
    height=280, margin=dict(t=30,b=40,l=60,r=20),
)
st.plotly_chart(fig_spec, use_container_width=True)

# ── Dominant frequencies ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_badge("Dominant Frequencies")
n_top = st.slider("Show top N frequencies", 5, 30, 10)
dominant = get_dominant_frequencies(fft_result, n=n_top)

# Top 5 metrics
cols_top = st.columns(min(5, n_top))
METRIC_COLORS = ["#7c3aed","#06d6a0","#ffd166","#ff6b6b","#00b4d8"]
for i, d in enumerate(dominant[:5]):
    cols_top[i].metric(
        f"#{i+1}",
        f"{d['frequency']:.1f} Hz",
        f"{d['magnitude_db']:.1f} dB"
    )

# Table
df = pd.DataFrame(dominant)
df.columns = ["Frequency (Hz)", "Magnitude (linear)", "Magnitude (dB)"]
st.dataframe(
    df.style.format({
        "Frequency (Hz)":     "{:.2f}",
        "Magnitude (linear)": "{:.6f}",
        "Magnitude (dB)":     "{:.2f}"
    }).background_gradient(
        subset=["Magnitude (dB)"],
        cmap="Purples"
    ),
    use_container_width=True,
    height=min(35 * n_top + 38, 400)
)

st.divider()

# ── Spectrogram ────────────────────────────────────────────────
section_badge("Time-Frequency — STFT Spectrogram")
st.markdown("""<p style="color:#7070a0;font-size:0.85rem;
font-family:'Space Grotesk',sans-serif;margin-bottom:0.75rem">
How the frequency content <em>evolves over time</em> — a static FFT cannot show this.
</p>""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    n_fft = st.select_slider("FFT window size",
                              options=[512, 1024, 2048, 4096], value=2048,
                              help="Larger → finer frequency, coarser time")
with col2:
    hop = st.select_slider("Hop length",
                            options=[128, 256, 512, 1024], value=512,
                            help="Smaller → finer time resolution")

with st.spinner("Computing spectrogram..."):
    stft_result = compute_stft_cached(audio.signal, audio.sample_rate, n_fft, hop)

fig_stft = plot_spectrogram(stft_result, max_hz=max_hz)
fig_stft.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,18,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#c4c4e0"),
    height=340, margin=dict(t=30,b=40,l=60,r=20),
)
st.plotly_chart(fig_stft, use_container_width=True)

col1, col2 = st.columns(2)
col1.caption(f"Frequency resolution: **{audio.sample_rate/n_fft:.1f} Hz/bin**")
col2.caption(f"Time resolution: **{hop/audio.sample_rate*1000:.1f} ms/frame**")
