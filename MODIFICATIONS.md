# Modifications log — rwkvjepa (everything lives in ts_decompose23d/)

Per user: do not modify TSL directly; canonical code is here, version-controlled. The trainer only
*imports* TSL's dataloader (read-only). Document every modification and every result.

## Setup (2026-06-25)
- **Self-contained harness** `train.py` — imports TSL `data_provider` (read-only), runs LOCAL models
  from `rwkvjepa/`, trains + evals ETTh1, prints `mse:`/`mae:`, appends `autoresearch_results.tsv`.
  Validates: GTR (cycle path) and cvjepa reproduce the run.py numbers (cvjepa 2ep→0.3985).
- **`rwkvjepa/` package** (canonical copies, local imports):
  - `video_jepa.py` (= VideoRWKVJEPA: RWKVMix/TemporalBlock/FrameEncoder/Predictor/build_frame_feats + Model)
  - `cometvideojepa.py` (CI lag-video motif-MoE; import fixed to `rwkvjepa.video_jepa`)
  - `gtr.py` (GTR seasonal: learnable global cycle queue Q + Conv2d fusion + MLP; self-contained)
  - `fused.py` (NEW: GTR-seasonal + lag-video motif-MoE)
- **Render switch** added to `cometvideojepa.py`: `--cm_render ai|human` — `ai` = raw-float lag-video
  (videos_ai, 1 value=1 pixel); `human` = 8-bit quantize+derender (videos). For the videos-vs-videos_ai test.
- **Fused model** `fused.py`, two modes (core idea: contribution must come from the VIDEO):
  - `fuse_deseason=1` (RCF, **video-primary**, default): learnable seasonal cycle Q is an additive bias,
    the **video models the deseasonalised series** → `fused = horizon_cycle + video(x − input_cycle)`.
  - `fuse_deseason=0` (GTR-primary boosting): `fused = GTR(x,cycle) + res_scale·video(x)`.

## Baseline selection (iter-2, run.py, 10 ep)
cvjepa_nojepa (lag, K=10, JEPA-off) = **0.388** (crosses linear bar 0.389). MoE helps (K1=0.416),
lag-video essential (raw-window=0.580), JEPA hurts (0.393), linres hurts (0.422). → baseline = lag-video
motif-MoE, JEPA-off.

## Results — see autoresearch_results.tsv + results_tsvideo.md. Round logs in .remember/tsv_*.log.

## FINAL (autoresearch, 4 rounds, 22 trainings)
Best **0.3789** ETTh1 h96 — fused video-primary RCF (d_model 128, lag-video motif-MoE, JEPA-off, videos_ai). Video-driven (cycle-only 1.006). Pushed to rwkvjepa.git.
