# Value diagnosis — TS-video representations (ETTh1 h96)

Linear-probe forecast MSE (z-space; ridge from flattened window-rep → future channels). persistence=1.283, raw-linear reference below; NLinear-with-RevIN≈0.389.

Lower probe MSE = more *linear forecasting value*. Lower bytes/value (from videos_ai/sizes.csv) = more compressible/redundant. The fusion should combine high-value + complementary formats.

| rank | format | probe MSE | probe MAE | mp4 B/value | raw:lossless |
|--:|---|--:|--:|--:|--:|
| 1 | raw | 0.5647 | 0.524 | - | - |
| 2 | splat_field | 0.629 | 0.5561 | 0.278 | 14.4 |
| 3 | chan_lag | 0.6664 | 0.5813 | 0.11 | 36.3 |
| 4 | radar_glyph | 1.0449 | 0.7785 | 0.257 | 15.5 |
| 5 | chan_scale | 1.1811 | 0.8477 | 0.79 | 5.1 |
| 6 | chan_corr | 1.6178 | 0.9759 | 1.031 | 3.9 |
| 7 | chan_gaf | 1.7488 | 1.0277 | 1.228 | 3.3 |
| 8 | chan_gram | 2.3843 | 1.2119 | 0.888 | 4.5 |
| 9 | chan_recur | 3.9342 | 1.3859 | 1.221 | 3.3 |

persistence baseline MSE = 1.2828

**Read:** linear encoders (raw/splat/scale/lag) should preserve the raw-linear value; nonlinear encoders (gaf/recur/gram/corr) that mangle channel values should score worse — confirming the gap is the forecasting pathway, not the pixels. Fuse the high-value linear reps with one complementary nonlinear rep and train VideoRWKVJEPA.
