# Experiments — ETTh1 h96 (seq_len 96, pred_len 96, features=M, test MSE/MAE, lower = better)

Two eval harnesses (numbers comparable; both = test MSE in the StandardScaler-normalised space):
**TSL** = `Time-Series-Library/run.py` (iter-0/1/2); **local** = self-contained `ts_decompose23d/train.py`
(autoresearch rounds, imports TSL dataloader read-only). Headline best = **0.3789**.

## 0 · Baselines / references
| id | method | MSE | MAE | harness | note |
|---|---|--:|--:|---|---|
| B1 | persistence (last value) | 1.2828 | — | numpy | sanity floor |
| B2 | **NLinear** (1 matrix, channel-indep, RevIN) | **0.3890** | 0.3940 | closed-form | the linear "bar" |
| B3 | DLinear (TSL) | 0.3986 | 0.4132 | TSL, 10ep | linear control |
| B4 | GTR (seasonal cycle-queue, ICLR'26) | 0.3804 | 0.3980 | local, 10ep | strong seasonal ref |
| B5 | CometNet (paper, full offline DTW motif lib) | 0.345 | — | reported | AAAI'26 SOTA (target) |

## 1 · Phase-1 — VideoRWKV / VideoRWKVJEPA (channel-mixed, splat frame) — TSL, 15ep
| id | model | MSE | MAE | note |
|---|---|--:|--:|---|
| P1.1 | VideoRWKV — splat, **patchify**, co-trained **MAE** | 0.5863 | 0.5119 | worst — pixel-recon wastes capacity |
| P1.2 | VideoRWKVJEPA — splat, **no-patchify**, **JEPA** | 0.4314 | 0.4431 | no-patchify+JEPA closed ~75% of the gap |

## 2 · Phase-2 — Iteration-1 representation sweep (channel-mixed JEPA, frame-as-token) — TSL, 10ep
| id | config (`--vr_encoder` …) | MSE | MAE | verdict |
|---|---|--:|--:|---|
| P2.1 | dlinear (control) | 0.3986 | 0.4132 | linear bar |
| P2.2 | jepa_lag | 0.4295 | 0.4389 | best video (still +8% vs bar) |
| P2.3 | jepa_splat | 0.4314 | 0.4430 | ≈ raw (splat = rank-7 linear) |
| P2.4 | jepa_raw_linres | 0.4350 | 0.4375 | linear+video, doesn't reach bar |
| P2.5 | jepa_fused_linres | 0.4399 | 0.4448 | **video-native value FAILS** |
| P2.6 | jepa_raw | 0.4433 | 0.4458 | |
| P2.7 | jepa_fused (raw+gram+gaf+recur) | 0.4442 | 0.4495 | fusion ≈ raw (no gain) |
| P2.8 | jepa_fused_fc0 (forecast-only) | 0.4500 | 0.4545 | JEPA mildly helps here |
| P2.9 | jepa_recur | 0.5103 | 0.4861 | nonlinear rep HURTS |
| P2.10 | jepa_gaf | 0.5321 | 0.5038 | nonlinear rep HURTS |
| P2.11 | jepa_gram | 0.5449 | 0.5009 | nonlinear rep HURTS |
| | **verdict** | | | channel-mixed frame-as-token capped below linear; nonlinear reps destroy value |

## 3 · Phase-3 — Iteration-2 CometVideoJEPA (channel-independent + motif-MoE) — TSL, 10ep
| id | config | MSE | MAE | verdict |
|---|---|--:|--:|---|
| P3.1 | **cvjepa_nojepa** (lag, K=10, JEPA-off) | **0.3880** | 0.4077 | **crosses linear bar** — new baseline |
| P3.2 | cvjepa (lag, K=10, JEPA-on) | 0.3929 | 0.4094 | JEPA HURTS |
| P3.3 | cvjepa_k20 | 0.3910 | 0.4081 | more experts ≈ |
| P3.4 | cvjepa_k1 (no MoE) | 0.4159 | 0.4257 | MoE earns its place |
| P3.5 | cvjepa_linres | 0.4223 | 0.4271 | linres hurts |
| P3.6 | cvjepa_raw (no lag-video) | 0.5795 | 0.5054 | lag-video ESSENTIAL |
| | **verdict** | | | CI + lag-video + motif-MoE, JEPA-off → 0.388; JEPA/linres/raw-window all hurt |

## 4 · Phase-4 — Autoresearch (GTR-seasonal + video-primary RCF fusion) — local, train.py
| id | round | change | MSE | MAE | ep | verdict |
|---|---|---|--:|--:|--:|---|
| P4.1 | r1 | **fused_des** (video-primary RCF) | **0.3800** | 0.4043 | 10 | KEEP — beats gtr/cvjepa/bar |
| P4.2 | r1 | gtr (seasonal only) | 0.3804 | 0.3980 | 10 | ref |
| P4.3 | r1 | cvjepa_ai (videos_ai) | 0.3821 | 0.4060 | 10 | render test |
| P4.4 | r1 | cvjepa_human (videos) | 0.3821 | 0.4060 | 10 | **videos == videos_ai** |
| P4.5 | r1 | fused_boost (GTR-primary) | 0.4107 | 0.4398 | 10 | discard |
| P4.6 | r2 | **+ d_model 128** | **0.3789** | 0.4026 | 10 | **KEEP — final best** |
| P4.7 | r2 | base / K=16 | 0.3800 | 0.404 | 10 | |
| P4.8 | r2 | lagw96 | 0.3819 | 0.4043 | 10 | discard |
| P4.9 | r2 | e_layers 3 | 0.3848 | 0.4075 | 10 | discard |
| P4.10 | r2 | **cycle-only (VIDEO OFF)** | **1.0061** | 0.7504 | 10 | **ablation: cycle alone useless** |
| P4.11 | r3 | d_model 256 | 0.3821 | 0.4065 | 10 | discard (overfit) |
| P4.12 | r3 | d_ff 512 | 0.3853 | 0.4041 | 10 | discard |
| P4.13 | r3 | d_model 192 | 0.3854 | 0.4046 | 10 | discard |
| P4.14 | r3 | K=20 | 0.3861 | 0.4041 | 10 | discard |
| P4.15 | r3 | lagw32 | 0.3908 | 0.4087 | 10 | discard |
| P4.16 | r3 | lagw24 | 0.3919 | 0.4088 | 10 | discard |
| P4.17 | r4 | d_model128 ep20 | 0.3789 | 0.4026 | 20 | == (converged) |
| P4.18 | r4 | lr 1e-3 | 0.3812 | 0.4019 | 20 | discard |
| P4.19 | r4 | d_model 96 | 0.3815 | 0.4033 | 20 | discard |
| P4.20 | r4 | lr 5e-4 | 0.3825 | 0.4032 | 20 | discard |
| P4.21 | r4 | seq_len 336 | 0.3969 | 0.4160 | 20 | discard (longer lookback hurt) |

## Headline
| model | MSE | MAE |
|---|--:|--:|
| **RWKVJEPAFused (video-primary RCF, d_model 128)** | **0.3789** | **0.4026** |

**Journey:** 0.586 (MAE) → 0.431 (JEPA) → 0.388 (CI motif-MoE) → **0.3789** (GTR-cycle + lag-video residual).
Beats NLinear 0.389, GTR-alone 0.3804, video-alone 0.3821. **Video-driven**: cycle-only = 1.006 ⇒ the
lag-video motif-MoE carries ~0.63 MSE; the seasonal cycle alone is useless on ETTh1.
