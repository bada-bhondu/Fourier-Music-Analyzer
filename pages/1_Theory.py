import streamlit as st
import numpy as np
from core.component_extractor import build_fourier_series_demo
from core.fourier_engine import benchmark_dft_vs_fft
from utils.plot_utils import plot_fourier_series_convergence, plot_benchmark
from utils.styles import inject_styles, section_badge, PLOTLY_LAYOUT, ACCENT_COLORS

inject_styles()

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.2rem">
    <span style="font-size:2rem">📐</span>
    <h1 style="margin:0">Fourier Theory</h1>
</div>
<p style="color:#7070a0;font-size:0.95rem;margin-top:0.1rem;margin-bottom:1.5rem;
font-family:'Space Grotesk',sans-serif">
The mathematical bridge from Continuous Fourier Series to the Discrete Fourier Transform.
</p>
""", unsafe_allow_html=True)


# ── Section 1: Fourier Series ──────────────────────────────────
section_badge("01 — Fourier Series")
st.markdown("""
Any periodic signal $f(t)$ with period $T$ can be expressed as an infinite
sum of sinusoids — the **Fourier Series**:
""")
st.latex(r"""
f(t) = \sum_{n=-\infty}^{\infty} c_n \, e^{\,j \, 2\pi n f_0 t}
\qquad \text{where} \quad f_0 = \frac{1}{T}
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown(r"""
**Complex Fourier coefficients** $c_n$:
$$c_n = \frac{1}{T} \int_0^T f(t)\, e^{-j\,2\pi n f_0 t}\, dt$$
""")
with col2:
    st.markdown("""
<div style="
    background:#12121e; border:1px solid rgba(124,58,237,0.2);
    border-radius:12px; padding:1rem 1.2rem; margin-top:0.5rem;
">
<p style="margin:0;color:#9090b8;font-size:0.88rem;
font-family:'Space Grotesk',sans-serif;line-height:1.7">
<span style="color:#7c3aed;font-weight:700">|cₙ|</span> — amplitude of the n-th harmonic<br>
<span style="color:#06d6a0;font-weight:700">∠cₙ</span> — phase offset<br>
<span style="color:#ffd166;font-weight:700">c₀</span> — DC offset (signal mean)<br>
<span style="color:#ff6b6b;font-weight:700">c₋ₙ = cₙ*</span> — conjugate symmetry for real signals
</p>
</div>
""", unsafe_allow_html=True)

# ── Convergence Demo ───────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section_badge("Convergence Demo")
st.markdown("Watch the Fourier series approximate a waveform term by term.")

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("""
    <div style="background:#12121e;border:1px solid rgba(124,58,237,0.2);
    border-radius:12px;padding:1rem;">
    """, unsafe_allow_html=True)
    waveform = st.selectbox("Waveform", ["square", "sawtooth", "triangle"],
                             label_visibility="collapsed")
    n_terms  = st.slider("Harmonics", 1, 20, 5)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    demo = build_fourier_series_demo(n_terms=n_terms, waveform=waveform)
    fig  = plot_fourier_series_convergence(demo)
    fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                         if k in ("paper_bgcolor","plot_bgcolor","font",
                                  "legend","margin")},
                      height=300)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div style="background:rgba(124,58,237,0.06);border-left:3px solid #7c3aed;
border-radius:0 10px 10px 0;padding:0.75rem 1.1rem;margin-top:-0.5rem">
<p style="margin:0;color:#9090b8;font-size:0.88rem;font-family:'Space Grotesk',sans-serif">
<strong style="color:#e2e2f0">Gibbs phenomenon:</strong>
The persistent overshoot near discontinuities is unavoidable with a finite number
of smooth sinusoids — a fundamental property of the Fourier basis.
</p>
</div>""", unsafe_allow_html=True)


# ── Section 2: Fourier Transform ──────────────────────────────
st.divider()
section_badge("02 — Fourier Transform")
st.markdown("""
For **aperiodic** signals (like a song), extend the period $T → ∞$.
The discrete sum becomes the **Fourier Transform**:
""")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Forward Transform**")
    st.latex(r"X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j\,2\pi f t}\, dt")
with col2:
    st.markdown("**Inverse Transform**")
    st.latex(r"x(t) = \int_{-\infty}^{\infty} X(f)\, e^{\,j\,2\pi f t}\, df")


# ── Section 3: Sampling & DFT ─────────────────────────────────
st.divider()
section_badge("03 — Discrete Fourier Transform")
st.markdown("""
Real audio is **sampled** at rate $f_s$, giving $N$ discrete samples.
The **DFT** is the exact discrete analogue:
""")
st.latex(r"""
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j\,2\pi k n / N}
\qquad f_k = \frac{k \cdot f_s}{N}
""")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div style="background:#12121e;border:1px solid rgba(124,58,237,0.2);
    border-radius:12px;padding:1rem;text-align:center">
    <p style="color:#7070a0;font-size:0.7rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.08em;margin:0 0 0.4rem 0;font-family:'Space Grotesk',sans-serif">
    Frequency Resolution</p>
    <p style="color:#7c3aed;font-size:1.3rem;font-weight:700;
    font-family:'JetBrains Mono',monospace;margin:0">Δf = fₛ / N</p>
    <p style="color:#6060a0;font-size:0.78rem;margin:0.4rem 0 0 0;
    font-family:'Space Grotesk',sans-serif">Finer resolution → longer recording</p>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div style="background:#12121e;border:1px solid rgba(6,214,160,0.2);
    border-radius:12px;padding:1rem;text-align:center">
    <p style="color:#7070a0;font-size:0.7rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.08em;margin:0 0 0.4rem 0;font-family:'Space Grotesk',sans-serif">
    Nyquist Limit</p>
    <p style="color:#06d6a0;font-size:1.3rem;font-weight:700;
    font-family:'JetBrains Mono',monospace;margin:0">f_max = fₛ / 2</p>
    <p style="color:#6060a0;font-size:0.78rem;margin:0.4rem 0 0 0;
    font-family:'Space Grotesk',sans-serif">Frequencies above alias</p>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div style="background:#12121e;border:1px solid rgba(255,209,102,0.2);
    border-radius:12px;padding:1rem;text-align:center">
    <p style="color:#7070a0;font-size:0.7rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.08em;margin:0 0 0.4rem 0;font-family:'Space Grotesk',sans-serif">
    CFS ↔ DFT Bridge</p>
    <p style="color:#ffd166;font-size:1.3rem;font-weight:700;
    font-family:'JetBrains Mono',monospace;margin:0">cₙ ≈ X[k] / N</p>
    <p style="color:#6060a0;font-size:0.78rem;margin:0.4rem 0 0 0;
    font-family:'Space Grotesk',sans-serif">Scaled DFT bins = series coefficients</p>
    </div>
    """, unsafe_allow_html=True)


# ── Section 4: Windowing ──────────────────────────────────────
st.divider()
section_badge("04 — Windowing & Spectral Leakage")
st.markdown(r"""
When the signal doesn't complete an integer number of cycles, sharp window edges
produce **spectral leakage** — spurious frequency content. Multiplying by a window
function $w[n]$ tapers edges to zero: $X_w[k] = \text{FFT}(x[n] \cdot w[n])$
""")

N_win = 256
t_win = np.linspace(0, 1, N_win)
freq  = 5.3
raw      = np.sin(2 * np.pi * freq * t_win)
windowed = raw * np.hanning(N_win)
raw_fft  = np.abs(np.fft.rfft(raw))     / N_win
win_fft  = np.abs(np.fft.rfft(windowed))/ N_win
f_axis   = np.fft.rfftfreq(N_win, 1/N_win)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig_leak = make_subplots(rows=1, cols=2,
    subplot_titles=["❌ Without Window (spectral leakage)",
                    "✅ Hann Window (energy concentrated)"])
for col_idx, (y, color) in enumerate(
        [(raw_fft, "#ff6b6b"), (win_fft, "#06d6a0")], start=1):
    fig_leak.add_trace(
        go.Bar(x=f_axis[:40], y=y[:40],
               marker_color=color,
               marker_line_width=0,
               opacity=0.85,
               showlegend=False),
        row=1, col=col_idx)
fig_leak.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,18,0.6)",
    font=dict(family="Space Grotesk, sans-serif", color="#c4c4e0", size=11),
    height=260, margin=dict(t=45, b=30, l=40, r=20),
    xaxis=dict(gridcolor="rgba(124,58,237,0.1)", title="Frequency bin"),
    xaxis2=dict(gridcolor="rgba(124,58,237,0.1)", title="Frequency bin"),
    yaxis=dict(gridcolor="rgba(124,58,237,0.1)"),
    yaxis2=dict(gridcolor="rgba(124,58,237,0.1)"),
)
st.plotly_chart(fig_leak, use_container_width=True)


# ── Section 5: FFT ────────────────────────────────────────────
st.divider()
section_badge("05 — Fast Fourier Transform")
st.markdown("""
Naïve DFT costs $O(N^2)$ multiplications. The **Cooley-Tukey FFT** (1965)
exploits twiddle-factor periodicity to recursively halve the problem:
""")
st.latex(r"""
X[k] = \underbrace{\sum_{n \text{ even}} x[n]\, e^{-j2\pi k n/N}}_{E[k]}
+\; e^{-j2\pi k/N} \underbrace{\sum_{n \text{ odd}} x[n]\, e^{-j2\pi k n/N}}_{O[k]}
\quad \Rightarrow \quad O(N \log N)
""")

c1, c2, c3 = st.columns(3)
for N_val, col, color in zip([512, 4096, 65536],
                              [c1, c2, c3],
                              ["#7c3aed", "#06d6a0", "#ffd166"]):
    fft_ops = int(N_val * np.log2(N_val))
    speedup = N_val ** 2 // fft_ops
    with col:
        st.markdown(f"""
        <div style="background:#12121e;border:1px solid {color}33;
        border-top:2px solid {color};border-radius:12px;
        padding:0.9rem 1rem;text-align:center">
        <p style="color:#7070a0;font-size:0.7rem;font-weight:700;
        text-transform:uppercase;letter-spacing:0.06em;margin:0 0 0.3rem;
        font-family:'Space Grotesk',sans-serif">N = {N_val:,}</p>
        <p style="color:{color};font-size:1.5rem;font-weight:800;
        font-family:'JetBrains Mono',monospace;margin:0">{speedup}× faster</p>
        <p style="color:#5a5a80;font-size:0.73rem;margin:0.3rem 0 0;
        font-family:'Space Grotesk',sans-serif">
        {N_val**2:,} → {fft_ops:,} ops</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
section_badge("Live Benchmark")
st.markdown("Compare actual DFT vs FFT execution time on your machine.")

seg_size = st.select_slider("Segment size (N)",
    options=[128, 256, 512, 1024, 2048], value=512)
_dummy = np.random.randn(seg_size)
bench  = benchmark_dft_vs_fft(_dummy, segment_size=seg_size)

col1, col2, col3 = st.columns(3)
col1.metric("Naive DFT", f"{bench['dft_ms']} ms")
col2.metric("FFT (scipy)", f"{bench['fft_ms']} ms")
col3.metric("Speedup", f"{bench['speedup']}×")

fig_bench = plot_benchmark(bench)
fig_bench.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items()
                            if k in ("paper_bgcolor","plot_bgcolor","font",
                                     "margin","legend")},
                         height=260)
st.plotly_chart(fig_bench, use_container_width=True)
