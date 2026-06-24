"""
decomp_lib.py — from-scratch time-series decomposition numerics (pure numpy/scipy/pywt).

We implement the methods that are NOT installed (PyEMD/vmdpy/pyts/pymssa/pydmd/ewtpy)
so we never touch the shared env (numpy-2 safe, deterministic, CPU only). Methods that
ARE installed (statsmodels STL/MSTL/UC/filters, pywt DWT/SWT/MRA/WPT/CWT, scipy spectral,
sklearn PCA) are called directly in the gallery — not re-implemented here.

Each function below has a synthetic self-test in __main__ with a known answer.

    modwt / imodwt   undecimated (MODWT) wavelet transform, equal-length per scale (causal core)
    emd / eemd / ceemdan   empirical mode decomposition family (sifting + noise-assisted)
    vmd              variational mode decomposition (Dragomiretskiy & Zosso 2014)
    ewt              empirical wavelet transform (Gilles 2013, Meyer partition-of-unity bank)
    ssa / mssa       singular spectrum analysis (uni / multivariate)
    dmd              dynamic mode decomposition (exact DMD)
    pca_cs           cross-sectional PCA (common vs idiosyncratic)
    mfdfa            multifractal detrended fluctuation analysis
    har_rv           realized-volatility HAR decomposition
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.interpolate import CubicSpline


# =====================================================================================
# MODWT — maximal-overlap (undecimated) DWT. Equal-length, shift-invariant coefficients.
#   Returns [n_levels + 1, N] = [W_1, ..., W_J, V_J]  (details coarsening + final smooth).
#   This is the shared "channel x scale at time t" engine for the video.
# =====================================================================================
def _modwt_filters(wavelet):
    w = pywt.Wavelet(wavelet)
    g = np.asarray(w.dec_lo, float) / np.sqrt(2.0)   # MODWT scaling filter
    h = np.asarray(w.dec_hi, float) / np.sqrt(2.0)   # MODWT wavelet filter
    return g, h


def _upsample(f, j):
    """Insert 2^(j-1)-1 zeros between filter taps (level-j MODWT filter)."""
    if j == 1:
        return f
    gap = 2 ** (j - 1)
    out = np.zeros((len(f) - 1) * gap + 1)
    out[::gap] = f
    return out


def _circ_conv(x, f):
    """Circular convolution: y[t] = sum_l f[l] x[(t-l) mod N]."""
    N = len(x)
    y = np.zeros(N)
    for l, fl in enumerate(f):
        if fl != 0.0:
            y += fl * np.roll(x, l)
    return y


def _circ_conv_adj(x, f):
    """Adjoint: y[t] = sum_l f[l] x[(t+l) mod N]."""
    N = len(x)
    y = np.zeros(N)
    for l, fl in enumerate(f):
        if fl != 0.0:
            y += fl * np.roll(x, -l)
    return y


def modwt(x, wavelet="db4", level=None):
    x = np.asarray(x, float)
    N = len(x)
    g, h = _modwt_filters(wavelet)
    if level is None:
        level = max(1, int(np.floor(np.log2(N))) - 1)
    W, V = [], x.copy()
    for j in range(1, level + 1):
        gj, hj = _upsample(g, j), _upsample(h, j)
        W.append(_circ_conv(V, hj))
        V = _circ_conv(V, gj)
    return np.vstack(W + [V])          # [level+1, N]


def imodwt(coeffs, wavelet="db4"):
    g, h = _modwt_filters(wavelet)
    level = coeffs.shape[0] - 1
    V = coeffs[-1].copy()
    for j in range(level, 0, -1):
        gj, hj = _upsample(g, j), _upsample(h, j)
        V = _circ_conv_adj(V, gj) + _circ_conv_adj(coeffs[j - 1], hj)
    return V


# =====================================================================================
# EMD family
# =====================================================================================
def _extrema(x):
    n = len(x)
    idx = np.arange(1, n - 1)
    gt = (x[idx] > x[idx - 1]) & (x[idx] >= x[idx + 1])
    lt = (x[idx] < x[idx - 1]) & (x[idx] <= x[idx + 1])
    return idx[gt], idx[lt]


def _envelope(t, ext_idx, x):
    """Cubic-spline envelope through extrema; anchor the two endpoints (boundary stability)."""
    tt = np.concatenate(([t[0]], t[ext_idx], [t[-1]]))
    yy = np.concatenate(([x[0]], x[ext_idx], [x[-1]]))
    tt, uniq = np.unique(tt, return_index=True)
    yy = yy[uniq]
    return CubicSpline(tt, yy)(t)


def emd(x, max_imf=10, max_sift=50, sd_thresh=0.2):
    """Empirical Mode Decomposition. Returns [n_imf + 1, N] (last row = residue/trend)."""
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float)
    res = x.copy()
    imfs = []
    for _ in range(max_imf):
        mx, mn = _extrema(res)
        if len(mx) + len(mn) < 3:
            break
        h = res.copy()
        for _s in range(max_sift):
            mx, mn = _extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                break
            up = _envelope(t, mx, h)
            lo = _envelope(t, mn, h)
            m = 0.5 * (up + lo)
            h_new = h - m
            sd = np.sum((h - h_new) ** 2) / (np.sum(h ** 2) + 1e-12)
            h = h_new
            if sd < sd_thresh:
                break
        imfs.append(h)
        res = res - h
        if np.all(np.abs(res) < 1e-12):
            break
    imfs.append(res)
    return np.array(imfs)


def _emd_first(x, max_sift=50, sd_thresh=0.2):
    return emd(x, max_imf=1, max_sift=max_sift, sd_thresh=sd_thresh)[0]


def _emd_fixed(x, K):
    """Decompose into exactly K detail slots + 1 residue (zero-pad missing slots)."""
    imfs = emd(x, max_imf=K)
    out = np.zeros((K + 1, len(x)))
    n = imfs.shape[0]
    nd = n - 1                                  # number of detail IMFs
    out[:min(nd, K)] = imfs[:min(nd, K)]
    out[K] = imfs[-1] + imfs[K:n - 1].sum(axis=0) if nd > K else imfs[-1]
    return out


def eemd(x, n_ens=50, noise_std=0.2, max_imf=8, seed=0):
    """Ensemble EMD: average IMFs over white-noise realisations (Wu & Huang 2009)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    sd = x.std() + 1e-12
    acc = np.zeros((max_imf + 1, len(x)))
    for _ in range(n_ens):
        acc += _emd_fixed(x + rng.normal(0, noise_std * sd, len(x)), max_imf)
    return acc / n_ens


def ceemdan(x, n_ens=40, noise_std=0.2, max_imf=8, seed=0):
    """Complete EEMD with Adaptive Noise (Torres et al. 2011). Additive by construction."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    N = len(x)
    white = [rng.normal(0, 1, N) for _ in range(n_ens)]
    Ew = [emd(w, max_imf=max_imf) for w in white]    # IMFs of each noise realisation

    def Ek(i, k):                                    # k-th IMF (1-indexed) of noise i
        m = Ew[i]
        return m[k - 1] if (k - 1) < (m.shape[0] - 1) else np.zeros(N)

    imfs = []
    sd = x.std() + 1e-12
    acc = np.zeros(N)
    for i in range(n_ens):
        acc += _emd_first(x + noise_std * sd * white[i])
    I1 = acc / n_ens
    imfs.append(I1)
    r = x - I1
    for k in range(1, max_imf):
        if r.std() < 1e-12:
            break
        sd = r.std()
        acc = np.zeros(N)
        for i in range(n_ens):
            acc += _emd_first(r + noise_std * sd * Ek(i, k + 1))
        Ik = acc / n_ens
        imfs.append(Ik)
        r = r - Ik
        mx, mn = _extrema(r)
        if len(mx) + len(mn) < 3:
            break
    imfs.append(r)
    return np.array(imfs)


# =====================================================================================
# VMD — variational mode decomposition (canonical Fourier-domain ADMM)
# =====================================================================================
def vmd(signal, alpha=2000.0, tau=0.0, K=3, DC=False, init=1, tol=1e-7, max_iter=500):
    """Returns (modes [K, N], center_freqs [K] in normalised [0,0.5])."""
    f = np.asarray(signal, float)
    saveT = len(f)
    fMirr = np.concatenate([f[:saveT // 2][::-1], f, f[saveT // 2:][::-1]])
    T = len(fMirr)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1.0 / T
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T // 2] = 0

    omega = np.zeros((max_iter, K))
    if init == 1:
        omega[0] = (0.5 / K) * np.arange(K)
    elif init == 2:
        omega[0] = np.sort(np.exp(np.log(1.0 / T) + (np.log(0.5) - np.log(1.0 / T)) * np.random.rand(K)))
    if DC:
        omega[0, 0] = 0.0

    u_hat_plus = np.zeros((max_iter, len(freqs), K), dtype=complex)
    lambda_hat = np.zeros((max_iter, len(freqs)), dtype=complex)
    uDiff = tol + np.spacing(1)
    n = 0
    sum_uk = np.zeros(len(freqs), dtype=complex)
    while uDiff > tol and n < max_iter - 1:
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        u_hat_plus[n + 1, :, 0] = (f_hat_plus - sum_uk - lambda_hat[n] / 2) / (1 + alpha * (freqs - omega[n, 0]) ** 2)
        if not DC:
            num = freqs[T // 2:] @ (np.abs(u_hat_plus[n + 1, T // 2:, 0]) ** 2)
            den = np.sum(np.abs(u_hat_plus[n + 1, T // 2:, 0]) ** 2) + 1e-300
            omega[n + 1, 0] = num / den
        for k in range(1, K):
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n] / 2) / (1 + alpha * (freqs - omega[n, k]) ** 2)
            num = freqs[T // 2:] @ (np.abs(u_hat_plus[n + 1, T // 2:, k]) ** 2)
            den = np.sum(np.abs(u_hat_plus[n + 1, T // 2:, k]) ** 2) + 1e-300
            omega[n + 1, k] = num / den
        lambda_hat[n + 1] = lambda_hat[n] + tau * (np.sum(u_hat_plus[n + 1], axis=1) - f_hat_plus)
        n += 1
        uDiff = np.spacing(1)
        for k in range(K):
            diff = u_hat_plus[n, :, k] - u_hat_plus[n - 1, :, k]
            uDiff += (1.0 / T) * np.vdot(diff, diff)
        uDiff = abs(uDiff)

    omega_final = omega[n]
    idxs = np.argsort(omega_final)
    u_hat = np.zeros((T, K), dtype=complex)
    u_hat[T // 2:T, :] = u_hat_plus[n, T // 2:T, :]
    u_hat[1:T // 2 + 1, :] = np.conj(u_hat_plus[n, T // 2:T, :][::-1])
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.zeros((K, len(t)))
    for k in range(K):
        u[k] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    u = u[:, T // 4:3 * T // 4]
    return u[idxs], omega_final[idxs]


# =====================================================================================
# EWT — empirical wavelet transform (Gilles 2013), Meyer partition-of-unity bank.
# =====================================================================================
def _ewt_boundaries(mag, N):
    """Detect up to N-1 boundaries: lowest minima between the N strongest spectral maxima."""
    mag = mag.copy()
    mag[0] = 0.0                                      # ignore DC
    # local maxima
    loc = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] >= mag[2:]))[0] + 1
    if len(loc) == 0:
        return np.array([], dtype=int)
    order = loc[np.argsort(mag[loc])[::-1]]
    peaks = np.sort(order[:N])
    bnds = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b > a + 1:
            bnds.append(a + 1 + int(np.argmin(mag[a + 1:b])))
        else:
            bnds.append((a + b) // 2)
    return np.array(sorted(set(bnds)), dtype=int)


def _beta(x):
    return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)


def _meyer_bank(af, w, gamma):
    """Partition-of-unity Meyer filter bank on |freq| in [0,pi].
    w = sorted internal boundaries in (0,pi); returns len(w)+1 filters with sum(sq)=1."""
    K = len(w) + 1
    bank = [np.zeros_like(af) for _ in range(K)]

    def trans(wb):                                   # raised-cosine arg around boundary wb
        return (af - (1 - gamma) * wb) / (2 * gamma * wb + 1e-30)

    # filter 0 : lowpass up to w[0]
    if K == 1:
        return [np.ones_like(af)]
    w0 = w[0]
    lp = np.zeros_like(af)
    lp[af <= (1 - gamma) * w0] = 1.0
    band = (af > (1 - gamma) * w0) & (af <= (1 + gamma) * w0)
    lp[band] = np.cos(np.pi / 2 * _beta(trans(w0)[band]))
    bank[0] = lp

    # middle bands
    for n in range(1, K - 1):
        wn, wn1 = w[n - 1], w[n]
        fb = np.zeros_like(af)
        flat = (af >= (1 + gamma) * wn) & (af <= (1 - gamma) * wn1)
        fb[flat] = 1.0
        rise = (af > (1 - gamma) * wn) & (af < (1 + gamma) * wn)
        fb[rise] = np.sin(np.pi / 2 * _beta(trans(wn)[rise]))
        fall = (af > (1 - gamma) * wn1) & (af < (1 + gamma) * wn1)
        fb[fall] = np.cos(np.pi / 2 * _beta(trans(wn1)[fall]))
        bank[n] = fb

    # last band : highpass above w[-1]
    wl = w[-1]
    hp = np.zeros_like(af)
    hp[af >= (1 + gamma) * wl] = 1.0
    rise = (af > (1 - gamma) * wl) & (af < (1 + gamma) * wl)
    hp[rise] = np.sin(np.pi / 2 * _beta(trans(wl)[rise]))
    bank[K - 1] = hp
    return bank


def ewt(f, N=4):
    """Empirical Wavelet Transform. Returns (modes [K, L], boundaries [rad])."""
    f = np.asarray(f, float)
    L = len(f)
    F = np.fft.fft(f)
    half = L // 2
    mag = np.abs(F[:half + 1])
    bidx = _ewt_boundaries(mag, N)
    if len(bidx) == 0:
        return f[None, :].copy(), np.array([])
    w = np.pi * bidx.astype(float) / half
    w = np.clip(w, 1e-6, np.pi - 1e-6)
    gamma = np.min([(w[i + 1] - w[i]) / (w[i + 1] + w[i]) for i in range(len(w) - 1)]) if len(w) > 1 else 0.5
    gamma = float(min(0.95 * gamma if len(w) > 1 else 0.5, w[0] / np.pi, (np.pi - w[-1]) / np.pi))
    gamma = max(gamma, 1e-3)
    freq = np.fft.fftfreq(L) * 2 * np.pi
    bank = _meyer_bank(np.abs(freq), w, gamma)
    # Tight-frame additive components: comp_k = ifft(|H_k|^2 F); sum_k |H_k|^2 = 1 -> sum = f.
    modes = np.array([np.real(np.fft.ifft(F * (fb ** 2))) for fb in bank])
    return modes, w


# =====================================================================================
# SSA / MSSA
# =====================================================================================
def _hankelize(M):
    """Diagonal (anti-diagonal) averaging of an L x K matrix -> series length L+K-1."""
    L, K = M.shape
    N = L + K - 1
    ii = (np.arange(L)[:, None] + np.arange(K)[None, :]).ravel()
    out = np.bincount(ii, weights=M.ravel(), minlength=N)
    cnt = np.bincount(ii, minlength=N)
    return out / cnt


def ssa(x, L, groups=None, n_keep=None):
    """Singular Spectrum Analysis. Returns (components [G, N], singular_values)."""
    x = np.asarray(x, float)
    N = len(x)
    K = N - L + 1
    traj = np.column_stack([x[i:i + L] for i in range(K)])      # L x K
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    d = len(s)
    if groups is None:
        nk = d if n_keep is None else min(n_keep, d)
        groups = [[i] for i in range(nk)]
    comps = []
    for g in groups:
        Xg = sum(s[i] * np.outer(U[:, i], Vt[i]) for i in g)
        comps.append(_hankelize(Xg))
    return np.array(comps), s


def mssa(X, L, n_keep=4):
    """Multivariate SSA (vertical block-Hankel stacking). Returns comps [n_keep, N, M], s."""
    X = np.asarray(X, float)
    N, M = X.shape
    K = N - L + 1
    blocks = [np.column_stack([X[i:i + L, m] for i in range(K)]) for m in range(M)]
    traj = np.vstack(blocks)                                     # (L*M) x K
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    n_keep = min(n_keep, len(s))
    comps = np.zeros((n_keep, N, M))
    for r in range(n_keep):
        Xr = s[r] * np.outer(U[:, r], Vt[r])                    # (L*M) x K
        for m in range(M):
            comps[r, :, m] = _hankelize(Xr[m * L:(m + 1) * L])
    return comps, s


# =====================================================================================
# DMD — exact dynamic mode decomposition
# =====================================================================================
def dmd(X, r=None, dt=1.0):
    """Exact DMD on X [features, time]. Returns dict(modes, eigs, omega, amplitudes, r)."""
    X = np.asarray(X, float)
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    if r is None:
        r = int(np.sum(s > 1e-10 * s[0]))
    r = max(1, min(r, len(s)))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    Atil = U.conj().T @ X2 @ Vt.conj().T @ np.diag(1.0 / s)
    lam, W = np.linalg.eig(Atil)
    Phi = X2 @ Vt.conj().T @ np.diag(1.0 / s) @ W
    omega = np.log(lam.astype(complex)) / dt
    b = np.linalg.lstsq(Phi, X[:, 0].astype(complex), rcond=None)[0]
    return dict(modes=Phi, eigs=lam, omega=omega, amplitudes=b, r=r)


def dmd_reconstruct(res, n):
    Phi, omega, b = res["modes"], res["omega"], res["amplitudes"]
    t = np.arange(n)
    dyn = (b[:, None] * np.exp(np.outer(omega, t)))             # [r, n]
    return np.real(Phi @ dyn)


# =====================================================================================
# Cross-sectional PCA (common vs idiosyncratic)
# =====================================================================================
def pca_cs(X, k=1):
    from sklearn.decomposition import PCA
    X = np.asarray(X, float)
    p = PCA(n_components=min(k, X.shape[1]))
    scores = p.fit_transform(X)
    common = p.inverse_transform(scores)
    return dict(common=common, idiosyncratic=X - common, components=p.components_,
                evr=p.explained_variance_ratio_, scores=scores)


# =====================================================================================
# MF-DFA
# =====================================================================================
def mfdfa(x, qs=None, scales=None, order=2):
    x = np.asarray(x, float)
    N = len(x)
    Y = np.cumsum(x - x.mean())
    if scales is None:
        scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(max(16, N // 4)), 18)).astype(int))
    if qs is None:
        qs = np.array([-5, -3, -2, -1, 1e-4, 1, 2, 3, 5], dtype=float)
    Fq = np.zeros((len(qs), len(scales)))
    for si, s in enumerate(scales):
        ns = N // s
        if ns < 1:
            Fq[:, si] = np.nan
            continue
        F2 = []
        t = np.arange(s)
        for v in range(ns):
            seg = Y[v * s:(v + 1) * s]
            seg2 = Y[N - (v + 1) * s:N - v * s]
            for sg in (seg, seg2):
                c = np.polyfit(t, sg, order)
                F2.append(np.mean((sg - np.polyval(c, t)) ** 2))
        F2 = np.asarray(F2)
        for qi, q in enumerate(qs):
            if abs(q) < 1e-3:
                Fq[qi, si] = np.exp(0.5 * np.mean(np.log(F2 + 1e-12)))
            else:
                Fq[qi, si] = (np.mean(F2 ** (q / 2.0))) ** (1.0 / q)
    logS = np.log(scales)
    hq = np.array([np.polyfit(logS, np.log(Fq[qi] + 1e-12), 1)[0] for qi in range(len(qs))])
    tau = qs * hq - 1
    alpha = np.gradient(tau, qs)
    falpha = qs * alpha - tau
    return dict(qs=qs, scales=scales, Fq=Fq, hq=hq, tau=tau, alpha=alpha, falpha=falpha)


# =====================================================================================
# HAR realized-volatility decomposition (Corsi 2009)
# =====================================================================================
def har_rv(series, day=24, week=7, month=30):
    r = np.diff(np.asarray(series, float))
    nd = len(r) // day
    RV = np.array([np.sum(r[d * day:(d + 1) * day] ** 2) for d in range(nd)])
    X, y = [], []
    for d in range(month, nd - 1):
        X.append([1.0, RV[d], RV[d - week + 1:d + 1].mean(), RV[d - month + 1:d + 1].mean()])
        y.append(RV[d + 1])
    X, y = np.asarray(X), np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return dict(RV=RV, beta=beta, pred=pred, y=y, r2=1 - ss_res / ss_tot, day=day)


# =====================================================================================
# Hamilton (2018) regression filter — the modern CAUSAL replacement for the HP filter.
#   trend_t = OLS fit of y_t on [1, y_{t-h}, ..., y_{t-h-p+1}];  cycle_t = y_t - trend_t.
#   Uses only the past (lags >= h) -> genuinely causal trend-cycle decomposition.
# =====================================================================================
def hamilton_filter(y, h=24, p=24):
    y = np.asarray(y, float)
    n = len(y)
    start = h + p - 1
    idx = np.arange(start, n)
    X = np.column_stack([np.ones(len(idx))] + [y[idx - h - j] for j in range(p)])
    yt = y[idx]
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    fit = X @ beta
    trend = np.full(n, np.nan)
    cycle = np.full(n, np.nan)
    trend[idx] = fit
    cycle[idx] = yt - fit
    return dict(trend=trend, cycle=cycle, beta=beta, start=start, h=h, p=p)


# =====================================================================================
# Bipower variation — continuous-vs-jump split of realized volatility (Barndorff-Nielsen
#   & Shephard 2004). RV = Σ r²;  BV = (π/2) Σ|r_i||r_{i-1}| (jump-robust);
#   jump J = max(RV-BV, 0);  continuous C = min(RV, BV). All trailing -> causal.
# =====================================================================================
def bipower_variation(series, day=24):
    r = np.diff(np.asarray(series, float))
    nd = len(r) // day
    RV = np.empty(nd)
    BV = np.empty(nd)
    for d in range(nd):
        rd = r[d * day:(d + 1) * day]
        RV[d] = np.sum(rd ** 2)
        BV[d] = (np.pi / 2.0) * np.sum(np.abs(rd[1:]) * np.abs(rd[:-1]))
    cont = np.minimum(RV, BV)
    jump = np.maximum(RV - BV, 0.0)
    return dict(RV=RV, BV=BV, cont=cont, jump=jump, day=day)


# =====================================================================================
# Self-tests
# =====================================================================================
def _zcr(x):
    return np.mean(np.abs(np.diff(np.sign(x))) > 0)


def _run_selftests():
    rng = np.random.default_rng(0)
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:14s} {detail}")

    # --- MODWT: energy preservation + perfect reconstruction
    x = rng.standard_normal(600)
    c = modwt(x, "db4", level=5)
    energy_ok = abs(np.sum(c ** 2) - np.sum(x ** 2)) / np.sum(x ** 2) < 1e-8
    recon_err = np.max(np.abs(imodwt(c, "db4") - x))
    check("modwt", energy_ok and recon_err < 1e-8,
          f"energyΔ={abs(np.sum(c**2)-np.sum(x**2)):.2e} recon={recon_err:.2e}")

    # --- EMD: exact reconstruction + high-freq IMF first
    t = np.linspace(0, 1, 1000)
    s = np.sin(2 * np.pi * 50 * t) + 0.6 * np.sin(2 * np.pi * 5 * t) + 0.2 * t
    im = emd(s, max_imf=8)
    rec = np.max(np.abs(im.sum(0) - s))
    freq_order = _zcr(im[0]) > _zcr(im[1])
    check("emd", rec < 1e-8 and freq_order, f"recon={rec:.2e} zcr0={_zcr(im[0]):.3f}>zcr1={_zcr(im[1]):.3f}")

    # --- EEMD: approximate reconstruction (noise-limited)
    ee = eemd(s, n_ens=30, noise_std=0.2, max_imf=6)
    rec = np.std(ee.sum(0) - s) / np.std(s)
    check("eemd", rec < 0.15, f"relReconStd={rec:.3f}")

    # --- CEEMDAN: exact additive reconstruction
    ce = ceemdan(s, n_ens=20, noise_std=0.2, max_imf=6)
    rec = np.max(np.abs(ce.sum(0) - s))
    check("ceemdan", rec < 1e-8 and ce.shape[0] >= 2, f"recon={rec:.2e} nmodes={ce.shape[0]}")

    # --- VMD: recover 3 tones
    N = 1000
    tt = np.arange(N)
    fs_true = np.array([0.02, 0.15, 0.40])
    sig = sum(np.cos(2 * np.pi * f * tt) for f in fs_true)
    u, om = vmd(sig, alpha=2000, K=3, tol=1e-7)
    om = np.sort(om)
    freq_err = np.max(np.abs(om - fs_true))
    rec = np.std(u.sum(0) - sig) / np.std(sig)
    check("vmd", freq_err < 0.02 and rec < 0.1, f"freqErr={freq_err:.4f} relRecon={rec:.3f} om={np.round(om,3)}")

    # --- EWT: tight-frame reconstruction
    em, w = ewt(sig, N=4)
    rec = np.max(np.abs(em.sum(0) - sig))
    check("ewt", rec < 1e-8 and em.shape[0] >= 2, f"recon={rec:.2e} nbands={em.shape[0]}")

    # --- SSA: full reconstruction + trend/sine separation
    tt = np.arange(400)
    xs = 0.01 * tt + 2 * np.sin(2 * np.pi * tt / 50) + 0.1 * rng.standard_normal(400)
    comps, sv = ssa(xs, L=80)
    rec = np.max(np.abs(comps.sum(0) - xs))
    trend_corr = np.corrcoef(comps[0], 0.01 * tt)[0, 1]
    check("ssa", rec < 1e-7 and abs(trend_corr) > 0.9, f"recon={rec:.2e} trendCorr={trend_corr:.3f}")

    # --- MSSA: full reconstruction across channels
    M = 3
    Xm = np.column_stack([2 * np.sin(2 * np.pi * tt / 50 + p) + 0.01 * tt for p in (0, 1, 2)])
    cm, sv = mssa(Xm, L=80, n_keep=10)
    rec = np.std(cm.sum(0) - Xm) / np.std(Xm)
    check("mssa", rec < 0.05, f"relRecon={rec:.3f}")

    # --- DMD: recover known eigenvalues of a genuine real linear system x_{k+1}=A x_k
    nf, T = 25, 220
    lam1, lam2 = 0.97 * np.exp(1j * 0.30), 0.85 * np.exp(1j * 0.90)

    def _rot(r, th):
        return r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    A_small = np.zeros((4, 4))
    A_small[:2, :2] = _rot(0.97, 0.30)        # eigs 0.97 e^{±i0.30}
    A_small[2:, 2:] = _rot(0.85, 0.90)        # eigs 0.85 e^{±i0.90}
    Q, _ = np.linalg.qr(rng.standard_normal((nf, 4)))   # orthonormal embedding
    s = rng.standard_normal(4)
    states = [s]
    for _ in range(T - 1):
        states.append(A_small @ states[-1])
    Xd = (Q @ np.array(states).T)            # [nf, T]
    res = dmd(Xd, r=4)
    e1 = np.min(np.abs(res["eigs"] - lam1))
    e2 = np.min(np.abs(res["eigs"] - lam2))
    rec = np.std(dmd_reconstruct(res, T) - Xd) / np.std(Xd)
    check("dmd", e1 < 0.05 and e2 < 0.05 and rec < 0.05, f"eigErr=({e1:.3f},{e2:.3f}) relRecon={rec:.3f}")

    # --- PCA cross-sectional: exact split
    Xc = rng.standard_normal((300, 5))
    pc = pca_cs(Xc, k=2)
    rec = np.max(np.abs(pc["common"] + pc["idiosyncratic"] - Xc))
    check("pca_cs", rec < 1e-9, f"splitErr={rec:.2e} evr={np.round(pc['evr'],3)}")

    # --- MF-DFA: white noise -> H(2)~0.5 ; random walk -> ~1.5
    wn = rng.standard_normal(4000)
    rw = np.cumsum(wn)
    m_wn = mfdfa(wn)
    m_rw = mfdfa(rw)
    h2_wn = m_wn["hq"][np.argmin(np.abs(m_wn["qs"] - 2))]
    h2_rw = m_rw["hq"][np.argmin(np.abs(m_rw["qs"] - 2))]
    check("mfdfa", 0.4 < h2_wn < 0.62 and 1.35 < h2_rw < 1.65, f"H2(wn)={h2_wn:.3f} H2(rw)={h2_rw:.3f}")

    # --- HAR-RV: runs, finite R2
    vol = np.abs(rng.standard_normal(24 * 200)) * (1 + 0.5 * np.sin(np.arange(24 * 200) / 200))
    h = har_rv(vol)
    check("har_rv", np.isfinite(h["r2"]) and len(h["beta"]) == 4, f"R2={h['r2']:.3f}")

    # --- Hamilton filter: removes a deterministic trend exactly; exact recon; stationary cycle
    tt = np.arange(1500)
    pure = 5 + 0.01 * tt                                        # deterministic linear trend
    cyc0 = np.nanmax(np.abs(hamilton_filter(pure, h=24, p=24)["cycle"]))
    y2 = 0.01 * tt + 1.5 * np.sin(2 * np.pi * tt / 60) + 0.05 * rng.standard_normal(1500)
    hf = hamilton_filter(y2, h=24, p=24)
    m = hf["start"]
    recon = np.nanmax(np.abs((hf["trend"] + hf["cycle"] - y2)[m:]))
    meanc = abs(np.nanmean(hf["cycle"][m:]))
    check("hamilton", cyc0 < 1e-6 and recon < 1e-9 and meanc < 0.2,
          f"pureTrendCyc={cyc0:.1e} recon={recon:.1e} meanCyc={meanc:.3f}")

    # --- Bipower variation: detects an injected jump
    base = 0.2 * rng.standard_normal(24 * 100)
    base[24 * 50 + 5] += 6.0                                    # one big jump on day 50
    bp = bipower_variation(base, day=24)
    jday = np.argmax(bp["jump"])
    check("bipower", jday == 50 and bp["jump"][50] > 5 * np.median(bp["jump"] + 1e-9),
          f"jumpDay={jday} J={bp['jump'][50]:.2f}")

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n[decomp_lib] {n_pass}/{len(results)} self-tests passed.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    print("[decomp_lib] running synthetic self-tests:")
    ok = _run_selftests()
    raise SystemExit(0 if ok else 1)
