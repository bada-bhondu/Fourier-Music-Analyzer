import streamlit as st
import numpy as np
import soundfile as sf
import io
from core.audio_loader import AudioData
from core.filter_engine import apply_filter_cached, audio_from_filter_result
from core.fourier_engine import compute_fft_cached
from core.reconstructor import _compute_metrics
from utils.plot_utils import plot_waveform, plot_spectrum
from utils.styles import inject_styles, section_badge

inject_styles()

st.markdown("""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.2rem">
    <span style="font-size:2rem">🎛️</span>
    <h1 style="margin:0">Interactive Filter & Playback</h1>
</div>
<p style="color:#7070a0;font-size:0.95rem;margin-top:0.1rem;margin-bottom:1.5rem;
font-family:'Space Grotesk',sans-serif">
Apply filters in real time — hear and see the effect on the signal instantly.
</p>
""", unsafe_allow_html=True)

audio: AudioData = st.session_state.get("audio", None)
if audio is None:
    st.markdown("""
    <div style="background:rgba(255,107,107,0.08);border:1px solid rgba(255,107,107,0.3);
    border-radius:12px;padding:1rem 1.5rem;display:flex;align-items:center;gap:0.75rem">
        <span style="font-size:1.3rem">⚠️</span>
        <span style="color:#ff6b6b;font-family:'Space Grotesk',sans-serif;font-size:0.9rem">
        No audio loaded — go to <strong>Upload & Analyze</strong> first.
        </span>
    </div>""", unsafe_allow_html=True)
    st.stop()


def _audio_bytes(signal, sr):
    buf = io.BytesIO()
    sf.write(buf, signal.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    buf.seek(0)
    return buf


def _chart_theme(fig, height=240, title=None):
    upd = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,18,0.6)",
        font=dict(family="Space Grotesk, sans-serif", color="#c4c4e0", size=11),
        height=height,
        margin=dict(t=35, b=35, l=55, r=15),
        xaxis=dict(gridcolor="rgba(124,58,237,0.1)"),
        yaxis=dict(gridcolor="rgba(124,58,237,0.1)"),
    )
    if title:
        upd["title"] = dict(text=title, font=dict(size=13, color="#c4c4e0"))
    fig.update_layout(**upd)
    return fig


# ── Filter config panel ────────────────────────────────────────
section_badge("Filter Configuration")

col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown("""
    <p style="color:#7070a0;font-size:0.75rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.08em;
    font-family:'Space Grotesk',sans-serif;margin-bottom:0.4rem">
    Filter Type
    </p>""", unsafe_allow_html=True)
    filter_type = st.radio(
        "Filter Type",
        ["Bandpass — keep a range",
         "Bandstop — remove a range",
         "Lowpass  — keep low freqs",
         "Highpass — keep high freqs",
         "Notch    — remove one frequency"],
        label_visibility="collapsed"
    )
    order = st.slider("Filter order", 2, 8, 4,
                       help="Higher order → steeper roll-off, more ringing")

    # Filter type explanation
    FILTER_HINTS = {
        "Bandpass": ("Keep only a frequency range. Good for isolating bass, midrange, or treble.", "#06d6a0"),
        "Bandstop": ("Remove a frequency range. Good for suppressing noise bands.", "#ff6b6b"),
        "Lowpass":  ("Pass low frequencies, cut highs. Classic 'muffled' sound.", "#7c3aed"),
        "Highpass": ("Pass high frequencies, cut lows. Removes bass rumble.", "#ffd166"),
        "Notch":    ("Surgically remove a single frequency. Classic use: 60 Hz hum removal.", "#00b4d8"),
    }
    for key, (hint, color) in FILTER_HINTS.items():
        if key in filter_type:
            st.markdown(f"""
            <div style="background:rgba(0,0,0,0.2);
            border-left:3px solid {color};border-radius:0 8px 8px 0;
            padding:0.6rem 0.8rem;margin-top:0.5rem">
            <p style="color:#9090b8;font-size:0.8rem;margin:0;
            font-family:'Space Grotesk',sans-serif">{hint}</p>
            </div>""", unsafe_allow_html=True)
            break

low = high = notch_hz = q_factor = None

with col_right:
    nyquist = audio.sample_rate // 2
    st.markdown("""
    <p style="color:#7070a0;font-size:0.75rem;font-weight:700;
    text-transform:uppercase;letter-spacing:0.08em;
    font-family:'Space Grotesk',sans-serif;margin-bottom:0.4rem">
    Frequency Parameters
    </p>""", unsafe_allow_html=True)

    if "Bandpass" in filter_type or "Bandstop" in filter_type:
        low  = st.slider("Low cutoff (Hz)",  20, nyquist-100, 300)
        high = st.slider("High cutoff (Hz)", 100, nyquist,    3000)
        if low >= high:
            st.error("Low must be less than high cutoff.")
            st.stop()
    elif "Lowpass" in filter_type:
        high = st.slider("Cutoff (Hz)", 100, nyquist, 4000)
    elif "Highpass" in filter_type:
        low = st.slider("Cutoff (Hz)", 20, nyquist-100, 300)
    else:
        notch_hz = st.slider("Notch frequency (Hz)", 20, nyquist, 60)
        q_factor = st.slider("Quality factor (Q)", 5, 60, 30,
                              help="Higher Q → narrower notch")
        bandwidth = notch_hz / (q_factor or 30)
        st.caption(f"Notch bandwidth: **{bandwidth:.1f} Hz** at {notch_hz} Hz")

# ── Apply filter ───────────────────────────────────────────────
with st.spinner("Applying filter..."):
    result = apply_filter_cached(
        audio.signal, audio.sample_rate,
        filter_type, order, low, high, notch_hz, q_factor or 30
    )

filtered_audio = audio_from_filter_result(result, audio)
difference     = audio.signal - result.filtered_signal

st.divider()

# ── Metrics ────────────────────────────────────────────────────
section_badge("Filter Effect Metrics")
metrics = _compute_metrics(audio.signal, result.filtered_signal)

c1, c2, c3, c4 = st.columns(4)
c1.metric("SNR",         f"{metrics['SNR_dB']:.2f} dB")
c2.metric("RMSE",        f"{metrics['RMSE']:.4f}")
c3.metric("Max Δ",       f"{metrics['MAX_ERROR']:.4f}")
c4.metric("Filter type", result.filter_type.title())

st.divider()

# ── Waveform comparison ────────────────────────────────────────
section_badge("Waveform Comparison")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""<p style="color:#00b4d8;font-weight:600;
    font-family:'Space Grotesk',sans-serif;font-size:0.9rem;margin-bottom:0.3rem">
    Original</p>""", unsafe_allow_html=True)
    fig = plot_waveform(audio, title="", color="#00b4d8")
    st.plotly_chart(_chart_theme(fig, 200), use_container_width=True)
    st.audio(_audio_bytes(audio.signal, audio.sample_rate), format="audio/wav")

with col2:
    st.markdown("""<p style="color:#06d6a0;font-weight:600;
    font-family:'Space Grotesk',sans-serif;font-size:0.9rem;margin-bottom:0.3rem">
    Filtered</p>""", unsafe_allow_html=True)
    fig = plot_waveform(filtered_audio, title="", color="#06d6a0")
    st.plotly_chart(_chart_theme(fig, 200), use_container_width=True)
    st.audio(_audio_bytes(result.filtered_signal, audio.sample_rate),
             format="audio/wav")

# ── Spectrum comparison ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_badge("Spectrum Comparison")

col1, col2 = st.columns(2)
with col1:
    fft_orig = compute_fft_cached(audio.signal, audio.sample_rate)
    fig = plot_spectrum(fft_orig, title="Original Spectrum")
    st.plotly_chart(_chart_theme(fig, 240, "Original Spectrum"),
                    use_container_width=True)

with col2:
    fft_filt = compute_fft_cached(result.filtered_signal, audio.sample_rate)
    fig = plot_spectrum(fft_filt, title="Filtered Spectrum")
    st.plotly_chart(_chart_theme(fig, 240, "Filtered Spectrum"),
                    use_container_width=True)

# ── Difference signal ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_badge("Removed Content")
st.markdown("""<p style="color:#7070a0;font-size:0.85rem;
font-family:'Space Grotesk',sans-serif;margin-bottom:0.5rem">
The component subtracted by the filter — what you can no longer hear.
</p>""", unsafe_allow_html=True)

diff_audio = AudioData(
    signal=difference.astype(np.float32),
    sample_rate=audio.sample_rate,
    duration=audio.duration,
    num_channels=audio.num_channels,
    bit_depth=audio.bit_depth,
    file_name="difference",
    file_format=audio.file_format,
)
fig = plot_waveform(diff_audio, title="", color="#ff6b6b")
st.plotly_chart(_chart_theme(fig, 200), use_container_width=True)
st.audio(_audio_bytes(difference, audio.sample_rate), format="audio/wav")
