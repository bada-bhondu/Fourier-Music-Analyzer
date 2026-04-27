import streamlit as st
import numpy as np
import soundfile as sf
import io
from core.audio_loader import AudioData
from core.filter_engine import (decompose_into_bands_cached,
                                 FREQUENCY_BANDS, audio_from_filter_result)
from core.component_extractor import (extract_peak_components_cached,
                                       extract_harmonic_series_cached,
                                       hpss_cached)
from core.fourier_engine import compute_fft_cached
from utils.plot_utils import (plot_band_waveforms, plot_band_spectra,
                               plot_harmonic_series, plot_waveform, plot_spectrum)
from utils.styles import inject_styles, section_badge

st.title("🔬 Signal Decomposition")
inject_styles()
audio: AudioData = st.session_state.get("audio", None)
if audio is None:
    st.warning("⚠️ No audio loaded. Go to **Upload & Analyze** first.")
    st.stop()

st.info(f"Analyzing: **{audio.file_name}** ({audio.duration:.1f}s, {audio.sample_rate} Hz)")

tab1, tab2, tab3 = st.tabs([
    "📊 Frequency Bands",
    "🎵 Tonal Components",
    "🥁 Harmonic / Percussive"
])

# ── Tab 1 ──────────────────────────────────────────────────
with tab1:
    section_badge("Psychoacoustic Band Decomposition")
    st.markdown("""
    The audio is split into 7 standard psychoacoustic bands using
    4th-order zero-phase Butterworth bandpass filters.
    """)

    with st.spinner("Applying bandpass filters..."):
        bands = decompose_into_bands_cached(audio.signal, audio.sample_rate)

    st.plotly_chart(plot_band_waveforms(bands, audio.sample_rate),
                    use_container_width=True)
    st.plotly_chart(plot_band_spectra(bands, audio.sample_rate),
                    use_container_width=True)

    section_badge("Per-Band Details")
    band_colors = ["#ff6b6b","#ffd166","#06d6a0",
                   "#00b4d8","#90e0ef","#a8dadc","#c77dff"]
    for i, (band_name, result) in enumerate(bands.items()):
        with st.expander(f"**{band_name}** "
                         f"({FREQUENCY_BANDS[band_name][0]}–"
                         f"{FREQUENCY_BANDS[band_name][1]} Hz)"):
            col1, col2, col3 = st.columns(3)
            sig = result.filtered_signal
            col1.metric("Peak Amplitude", f"{np.max(np.abs(sig)):.4f}")
            col2.metric("RMS Energy",     f"{np.sqrt(np.mean(sig**2)):.4f}")
            col3.metric("Samples",        f"{len(sig):,}")

            st.plotly_chart(
                plot_waveform(
                    audio_from_filter_result(result, audio),
                    title=f"{band_name} Waveform",
                    color=band_colors[i]
                ),
                use_container_width=True
            )
            buf = io.BytesIO()
            sf.write(buf, sig, audio.sample_rate, format="WAV", subtype="FLOAT")
            buf.seek(0)
            st.audio(buf, format="audio/wav")

# ── Tab 2 ──────────────────────────────────────────────────
with tab2:
    section_badge("Peak Tonal Component Extraction")
    st.markdown(r"""
    Identifies the $N$ most prominent spectral peaks and reconstructs
    each as a pure sinusoid: $x_k(t) = A_k \cos(2\pi f_k t + \varphi_k)$
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_comp = st.slider("Number of components", 5, 50, 15)
    with col2:
        min_dist = st.slider("Min separation (Hz)", 1, 50, 10)
    with col3:
        window = st.selectbox("Window", ["hann", "hamming", "blackman"])

    with st.spinner("Extracting tonal components..."):
        decomp = extract_peak_components_cached(
            audio.signal, audio.sample_rate,
            n_comp, min_dist, window
        )

    st.metric("Signal Coverage", f"{decomp.coverage_db:.2f} dB",
              help="How much of total signal energy these components capture")

    import pandas as pd
    rows = [{
        "Frequency (Hz)": f"{c.frequency:.2f}",
        "Magnitude (dB)": f"{c.magnitude_db:.2f}",
        "Phase (rad)":    f"{c.phase:.3f}",
        "Bin Index":      c.bin_index
    } for c in decomp.components]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    section_badge("Component Waveforms")
    selected = st.multiselect(
        "Select components to visualize",
        options=[f"{c.frequency:.1f} Hz" for c in decomp.components],
        default=[f"{c.frequency:.1f} Hz" for c in decomp.components[:3]]
    )
    for c in decomp.components:
        label = f"{c.frequency:.1f} Hz"
        if label in selected:
            dummy = AudioData(
                signal=c.signal, sample_rate=audio.sample_rate,
                duration=audio.duration, num_channels=1,
                bit_depth=None, file_name=label, file_format=""
            )
            st.plotly_chart(
                plot_waveform(dummy, title=f"Component: {label} | {c.magnitude_db:.1f} dB"),
                use_container_width=True
            )

    section_badge("Harmonic Series Analysis")
    f0 = st.number_input("Fundamental frequency f₀ (Hz)",
                          min_value=20.0, max_value=2000.0,
                          value=float(decomp.components[0].frequency), step=0.5)
    n_harm = st.slider("Number of harmonics", 2, 16, 8)

    with st.spinner("Extracting harmonic series..."):
        harm_series = extract_harmonic_series_cached(
            audio.signal, audio.sample_rate, f0, n_harm
        )

    st.plotly_chart(plot_harmonic_series(harm_series), use_container_width=True)

    buf = io.BytesIO()
    sf.write(buf, harm_series.combined_signal,
             audio.sample_rate, format="WAV", subtype="FLOAT")
    buf.seek(0)
    st.caption("▶ Play reconstructed harmonic series:")
    st.audio(buf, format="audio/wav")

# ── Tab 3 ──────────────────────────────────────────────────
with tab3:
    section_badge("Harmonic-Percussive Source Separation")
    st.markdown("""
    Separates into **Harmonic** (sustained tones) and **Percussive** (transients)
    via median filtering in the spectrogram domain.
    """)
    st.warning("⚠️ First run takes 30–60s on long files. Result is cached after.")

    kernel = st.slider("Median filter kernel size", 11, 61, 31, step=2)

    with st.spinner("Running HPSS..."):
        h_sig, p_sig = hpss_cached(audio.signal, audio.sample_rate, kernel)

    hpss_harmonic = AudioData(
        signal=h_sig, sample_rate=audio.sample_rate,
        duration=len(h_sig)/audio.sample_rate,
        num_channels=audio.num_channels, bit_depth=audio.bit_depth,
        file_name=f"{audio.file_name} [harmonic]",
        file_format=audio.file_format
    )
    hpss_percussive = AudioData(
        signal=p_sig, sample_rate=audio.sample_rate,
        duration=len(p_sig)/audio.sample_rate,
        num_channels=audio.num_channels, bit_depth=audio.bit_depth,
        file_name=f"{audio.file_name} [percussive]",
        file_format=audio.file_format
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎵 Harmonic Component")
        st.plotly_chart(plot_waveform(hpss_harmonic, title="Harmonic",
                                      color="#00b4d8"), use_container_width=True)
        fft_h = compute_fft_cached(h_sig, audio.sample_rate)
        st.plotly_chart(plot_spectrum(fft_h, title="Harmonic Spectrum"),
                        use_container_width=True)
        buf = io.BytesIO()
        sf.write(buf, h_sig, audio.sample_rate, format="WAV", subtype="FLOAT")
        buf.seek(0)
        st.audio(buf, format="audio/wav")

    with col2:
        st.markdown("#### 🥁 Percussive Component")
        st.plotly_chart(plot_waveform(hpss_percussive, title="Percussive",
                                      color="#ff6b6b"), use_container_width=True)
        fft_p = compute_fft_cached(p_sig, audio.sample_rate)
        st.plotly_chart(plot_spectrum(fft_p, title="Percussive Spectrum"),
                        use_container_width=True)
        buf = io.BytesIO()
        sf.write(buf, p_sig, audio.sample_rate, format="WAV", subtype="FLOAT")
        buf.seek(0)
        st.audio(buf, format="audio/wav")
