import streamlit as st
import numpy as np
import soundfile as sf
import io
from core.audio_loader import AudioData
from core.reconstructor import (reconstruct_full_cached, reconstruct_topk_cached,
                                  reconstruct_band_cached, snr_vs_components_cached)
from utils.plot_utils import plot_reconstruction_comparison, plot_snr_curve
from utils.styles import inject_styles, section_badge, PLOTLY_LAYOUT

inject_styles()

st.markdown("""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.2rem">
    <span style="font-size:2rem">🔁</span>
    <h1 style="margin:0">Reconstruct & Error Analysis</h1>
</div>
<p style="color:#7070a0;font-size:0.95rem;margin-top:0.1rem;margin-bottom:1.5rem;
font-family:'Space Grotesk',sans-serif">
IFFT round-trip, top-k sparse reconstruction, and quantitative error metrics.
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


def _apply_chart_theme(fig, height=300):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,18,0.6)",
        font=dict(family="Space Grotesk, sans-serif", color="#c4c4e0"),
        height=height, margin=dict(t=35,b=40,l=55,r=20),
        legend=dict(bgcolor="rgba(18,18,30,0.9)",
                    bordercolor="rgba(124,58,237,0.3)",borderwidth=1),
        xaxis=dict(gridcolor="rgba(124,58,237,0.1)"),
        yaxis=dict(gridcolor="rgba(124,58,237,0.1)"),
    )
    return fig


def _metric_row(metrics, inf_snr=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE",       f"{metrics['MSE']:.2e}")
    c2.metric("RMSE",      f"{metrics['RMSE']:.2e}")
    c3.metric("SNR",       "∞ dB ✨" if inf_snr else f"{metrics['SNR_dB']:.1f} dB")
    c4.metric("Max Error", f"{metrics['MAX_ERROR']:.2e}")


tab1, tab2, tab3 = st.tabs([
    "  ✅  Full Reconstruction  ",
    "  📉  Top-k Partial  ",
    "  🎛️  Band Reconstruction  ",
])


# ── Tab 1: Full reconstruction ─────────────────────────────────
with tab1:
    section_badge("Full IFFT Round-Trip")
    st.markdown("""<p style="color:#7070a0;font-size:0.87rem;
    font-family:'Space Grotesk',sans-serif;margin-bottom:0.75rem">
    Complete FFT → IFFT pipeline. Near-perfect reconstruction confirms the
    pipeline is numerically lossless.
    </p>""", unsafe_allow_html=True)

    window = st.selectbox("Window function",
                           ["hann", "hamming", "blackman", "none"], key="w_full")

    with st.spinner("Reconstructing..."):
        full = reconstruct_full_cached(audio.signal, audio.sample_rate, window)

    _metric_row(full.metrics, inf_snr=np.isinf(full.metrics["SNR_dB"]))

    fig = plot_reconstruction_comparison(
        full.original, full.reconstructed, full.error_signal, audio.sample_rate)
    st.plotly_chart(_apply_chart_theme(fig, 340), use_container_width=True)

    st.markdown('<p style="color:#5a5a80;font-size:0.78rem;font-style:italic;'
                'font-family:Space Grotesk,sans-serif">▶ Reconstructed audio</p>',
                unsafe_allow_html=True)
    st.audio(_audio_bytes(full.reconstructed, audio.sample_rate),
             format="audio/wav")


# ── Tab 2: Top-k partial ───────────────────────────────────────
with tab2:
    section_badge("Top-k Sparse Reconstruction")
    st.markdown("""<p style="color:#7070a0;font-size:0.87rem;
    font-family:'Space Grotesk',sans-serif;margin-bottom:0.75rem">
    Keep only the <em>k</em> largest FFT bins. As <em>k</em> grows, SNR rises —
    demonstrating the signal sparsity underlying audio compression.
    </p>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        k = st.select_slider(
            "Components to keep (k)",
            options=[10, 50, 100, 500, 1000, 5000,
                     10000, 50000, 100000, 500000],
            value=1000
        )
        total_bins = len(audio.signal) // 2 + 1
        pct = 100 * k / total_bins
        st.markdown(f"""
        <div style="background:#12121e;border:1px solid rgba(124,58,237,0.2);
        border-radius:12px;padding:0.9rem;margin-top:0.5rem">
        <p style="color:#7070a0;font-size:0.7rem;font-weight:700;
        text-transform:uppercase;letter-spacing:0.08em;margin:0 0 0.5rem;
        font-family:'Space Grotesk',sans-serif">Sparsity</p>
        <p style="color:#e2e2f0;font-size:1.3rem;font-weight:700;
        font-family:'JetBrains Mono',monospace;margin:0">
        {k:,} / {total_bins:,}</p>
        <div style="background:#1a1a2e;border-radius:6px;height:6px;
        margin:0.5rem 0;overflow:hidden">
            <div style="background:linear-gradient(90deg,#7c3aed,#06d6a0);
            height:100%;width:{min(pct,100):.2f}%;border-radius:6px"></div>
        </div>
        <p style="color:#7070a0;font-size:0.78rem;margin:0;
        font-family:'JetBrains Mono',monospace">{pct:.3f}% of spectrum</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        with st.spinner("Reconstructing with top-k..."):
            topk = reconstruct_topk_cached(audio.signal, audio.sample_rate, k)
        c1, c2, c3 = st.columns(3)
        c1.metric("SNR",  f"{topk.metrics['SNR_dB']:.2f} dB")
        c2.metric("RMSE", f"{topk.metrics['RMSE']:.4f}")
        c3.metric("PSNR", f"{topk.metrics['PSNR_dB']:.2f} dB")

    fig = plot_reconstruction_comparison(
        topk.original, topk.reconstructed, topk.error_signal, audio.sample_rate)
    st.plotly_chart(_apply_chart_theme(fig, 320), use_container_width=True)

    # SNR sweep
    st.markdown("<br>", unsafe_allow_html=True)
    section_badge("SNR vs Components Sweep")
    with st.spinner("Running sweep..."):
        sweep = snr_vs_components_cached(
            audio.signal, audio.sample_rate,
            k_values=tuple([100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000])
        )
    fig_snr = plot_snr_curve(sweep)
    st.plotly_chart(_apply_chart_theme(fig_snr, 300), use_container_width=True)
    st.caption(
        "Log-scale x-axis. The slowly rising curve reflects that real music "
        "energy is spread across all bins — unlike pure tones."
    )

    st.markdown('<p style="color:#5a5a80;font-size:0.78rem;font-style:italic;'
                'font-family:Space Grotesk,sans-serif">▶ Reconstructed audio '
                f'(k={k:,} components)</p>', unsafe_allow_html=True)
    st.audio(_audio_bytes(topk.reconstructed, audio.sample_rate),
             format="audio/wav")


# ── Tab 3: Band reconstruction ─────────────────────────────────
with tab3:
    section_badge("Frequency Band Reconstruction")
    st.markdown("""<p style="color:#7070a0;font-size:0.87rem;
    font-family:'Space Grotesk',sans-serif;margin-bottom:0.75rem">
    Zero out all bins outside the chosen band. The ideal frequency-domain
    bandpass — compare with the time-domain filters on the Filter page.
    </p>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        low_hz  = st.slider("Low cutoff (Hz)",  20,    2000,  20)
    with col2:
        high_hz = st.slider("High cutoff (Hz)", 100, 22000, 20000)

    if low_hz >= high_hz:
        st.error("Low cutoff must be less than high cutoff.")
        st.stop()

    with st.spinner("Reconstructing band..."):
        band_r = reconstruct_band_cached(
            audio.signal, audio.sample_rate, low_hz, high_hz)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Band",      f"{low_hz}–{high_hz} Hz")
    c2.metric("Bins kept", f"{band_r.num_components:,}")
    c3.metric("SNR",       f"{band_r.metrics['SNR_dB']:.2f} dB")
    c4.metric("RMSE",      f"{band_r.metrics['RMSE']:.4f}")

    fig = plot_reconstruction_comparison(
        band_r.original, band_r.reconstructed, band_r.error_signal, audio.sample_rate)
    st.plotly_chart(_apply_chart_theme(fig, 320), use_container_width=True)

    st.markdown(f'<p style="color:#5a5a80;font-size:0.78rem;font-style:italic;'
                f'font-family:Space Grotesk,sans-serif">▶ Band audio '
                f'({low_hz}–{high_hz} Hz)</p>', unsafe_allow_html=True)
    st.audio(_audio_bytes(band_r.reconstructed, audio.sample_rate),
             format="audio/wav")
