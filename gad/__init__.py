"""GAD: a guarded adaptive anomaly detector for industrial control systems.

Seven layers, L0 to L6: input quality (L0), two predictors (L1a from the other
sensors, L1b from physical laws), residual scaling (L2), CUSUM evidence (L3),
the decision (L4), guarded adaptation (L5) and an evidence record per alarm
(L6). `gad.commission.commission` fits it once on clean data; `gad.settings`
holds the configuration evaluated in the paper.
"""
