# Claim matrix

Every quantitative claim of the camera-ready paper (`paper/camera-ready.pdf`, labels CR)
and the headline claims and tables of the extended version (`paper/extended.pdf`, labels
EXT), with the result file and key that hold it and the command that produces the file.
Section, table and figure numbers are those of the current PDFs. `scripts/check-claims.py`
checks each row automatically (214 checks in total) after `lexi/check_results.py` (97
checks on the same files).

Result files are in `results/`. Commands run from `lexi/`:

| Short name | File | Command | Time |
| :-- | :-- | :-- | :-- |
| core | `core_results.json` | `python3 experiments_core.py` | 10 to 15 min |
| ext | `extended_results.json` | `python3 extended_experiments.py` | 27 s |
| dec | `decision_checks.json` | `python3 decision_checks.py` | 19 s |
| real | `real_data_results.json` | `python3 real_data_experiment.py` | 48 s |
| rob | `robustness_results.json` | `scripts/run-extended.sh` (run_incremental sequence) | 7 min |
| dep | `deployment_results.json` | `python3 deployment_experiments.py` | 3 min |
| sim | `lexi/sim.py`, `lexi/baselines.py`, `lexi/controller.py` | configuration in code | none |

Node ids: E0 = `edge_A`, E1 = `edge_B`, E2 = `edge_C`, C0 = `cloud_A`, C2 = `cloud_C`,
S1 = `srvl_B`, S2 = `srvl_C`. Method keys: `Lexi-strict` = Lexi, `constrained_rl` =
Constr-WS, `ws_tuned` = WS-tuned, `ws_default` = WS-default.

## Camera-ready

### Abstract and Sect. 1

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| 15-seed simulation | 15 | core `e1_satisfaction.n_seeds` |
| Lexi 1.46% SLO violations, PII exposure 0.05 | 1.458, 0.054 | core `e1_satisfaction.table.Lexi-strict` |
| Untuned sum 23% SLO, 1.94 PII | 23.025, 1.939 | core `e1_satisfaction.table.ws_default` |
| One of 84 weightings meets the target | 84 weightings, fraction 0.012 | core `e2_weight_sweep` |
| Privacy under-scaled 10x: tuned 1.85, constrained 1.84 of two stages, Lexi unchanged | 1.848, 1.837, 0.052 at every factor | core `e3_scale_invariance.objectives.privacy` (factor 0.1), ext `B_constrrl_scale.rows` |

### Sect. 2 and Fig. 1

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| Hinge scale rho = 60 ms, plateau at 320 ms | 60, 260 + 60 | sim `SLO_MARGIN`, `Continuum.slo` |
| Evaluated rule is strict (eta = 0) | `STRICT_ETA = 0` | `experiments_core.py` |
| Def. 2 final tie on cost | the evaluated controller applies a stability hold first. It changes 0 of 24,000 main-run decisions, and Sect. 5 now discloses it (row Lex-LP below). **Resolved** | dec `hysteresis_effect.main`, core `lexlp_agreement` |
| Fig. 1 counts 40, 11, 4, 1, 1 | 40, 11, 4, 1 (cost level returns 1) | ext `D_worked_example.band_trace` |

### Sect. 3 and Table 1 (worked decision, seed 0, step 904)

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| Origin region 2, load 0.765, t = 904 | 2, 0.765, 904 | ext `D_worked_example.request` |
| 40 candidates, 39 never executed | 40 | ext `D_worked_example.n_cand` |
| Per-node models fitted on steps 0 to 903 (corrected from "1 to 903") | decision at step 904 after 904 updates. **Resolved** | ext `D_worked_example.step`, `extended_experiments.worked_example` |
| Level 0: 40 to 11 | 11 survivors | ext `band_trace[0]` |
| 9 of the 11 truly meet the 260 ms SLO, #24 and one other admitted by error | 9, admitted #24 and #33 | ext `D_worked_example.panel` (`true_lat_ms`) |
| Level 1: 11 to 4, {2, 19, 20, 21} | same | ext `band_trace[1]` |
| #21 wins carbon outright, cost tie-break unused | level-2 survivors [21] | ext `band_trace[2]`, `chosen` |
| Band eta2 = 0.08 keeps #20 (0.387 <= 0.314 + 0.08), cost picks #21 | survivors [20, 21], choice 21 | dec `worked_decision.carbon_band_0.08_*` |
| Untuned sum picks #6 | 6 | dec `worked_decision.ws_default_choice` |
| Tuned sum 0.030 vs 0.111 for #6, picks #21 | 0.0296, 0.1108, 21 | dec `worked_decision.ws_tuned_*` |
| Constr-WS picks #21 | 21 | dec `worked_decision.constr_ws_choice` |
| 10x privacy under-scaling switches the tuned sum to #6 (0.021 vs 0.030) | 6, 0.0208, 0.0296 | dec `worked_decision.ws_tuned_privacy_x0.1_*` |
| No rescaling changes the lexicographic choice | true for factors 0.1 to 30 | dec `worked_decision.strict_choice_invariant_to_privacy_rescaling` |
| Table 1 rows #2, #6, #19, #20, #21, #24: placement, predicted and true latency, g0 to g3, elimination level | all match | ext `D_worked_example.panel`, `band_trace`. Render: `python3 make_tables.py camera` |

### Sect. 5, Sect. 5.1 and Table 2

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| M = 7 nodes over 3 regions, K = 6 stages, 260 ms SLO, 40-candidate menu | same | sim `build_nodes`, `build_dag`, core `_meta.cand_cap` |
| 15 seeds, steps 800 to 1599 | warmup 800, horizon 1600 | core `_meta` |
| Sweeps use 6 to 8 seeds | 6 (weight sweep), 8 (scale, menu, SLO sweep, config B) | core `e2`, `e3`, `e5`, `e6` `n_seeds`, ext `A_menu`, `B_constrrl_scale` |
| Table 2, seven rows (b, RTT in/out, q, cost, energy, mean intensity, privacy) | all 70 values match | sim `build_nodes`. Render: `python3 make_tables.py camera` |
| DAG demands 8, 16, 30, 25, 20, 18, PII on auth and pay | same | sim `build_dag` |
| Intensity amplitude 60 to 120 gCO2/kWh, 240-step cycle | 60 to 120, `DAY = 240` | sim `build_nodes`, `DAY` |
| Menu: 22 structured and 18 seeded random placements | 22, 18 | `controller.candidate_set` |
| WS-default (0.30, 0.12, 0.30, 0.28), WS-tuned (0.85, 0.10, 0.00, 0.05) | same | `baselines.DEFAULT_WS_W`, `TUNED_WS_W` |
| WS-tuned better than the best of the 84-point sweep | 3.28% vs 6.67% SLO | core `e3_scale_invariance` (factor 1), `e2_weight_sweep.best_ws` |
| Constr-WS minimises an equal-weight sum of the rest | (0.34, 0.33, 0.33) | `baselines.ConstrainedWeightedSum.W_REST` |
| Lex-LP, the same rule without Lexi's hold on a surviving previous placement, matches Lexi on all 24,000 decisions (added in the current PDF) | agreement 1.00 over 24,000 decisions, per-metric gap 0. **Resolved** | core `lexlp_agreement`, `e1_satisfaction.lexlp_vs_strict_max_gap`, dec `hysteresis_effect.main` |

### Table 3 (15 seeds)

| Row | Paper (SLO%, PII, carbon, cost, churn, Inv%) | Artifact | Source |
| :-- | :-- | :-- | :-- |
| Lexi | 1.46, 0.05, 0.98, 3.21, 1.35, 0.0 | 1.458, 0.054, 0.983, 3.209, 1.345, 0.0 | core `e1_satisfaction.table`, `e1_inversion.inversion` |
| Constr-WS | 1.32, 0.05, 0.98, 3.19, 1.32, 0.7 | 1.317, 0.054, 0.984, 3.193, 1.316, 0.683 | same |
| WS-tuned | 3.19, 0.01, 0.99, 3.20, 1.28, 3.7 | 3.192, 0.013, 0.985, 3.202, 1.284, 3.65 | same |
| WS-default | 23.02, 1.94, 0.45, 1.22, 0.65, 94.7 | 23.025, 1.939, 0.453, 1.217, 0.654, 94.725 | same |

Render: `python3 make_tables.py camera`.

### Fig. 2

| Panel | Data | Source | Render |
| :-- | :-- | :-- | :-- |
| (a) 84 weightings, 27 above 25%, target SLO <= 8% and PII <= 0.1, Lexi at (0.052, 1.42), 6 seeds | 57 plotted + 27 counted | core `e2_weight_sweep` | `python3 make_camera_figures.py` |
| (b) privacy-scale factors 0.1 to 30, Lexi, Constr-WS, WS-tuned, 8 seeds | see EXT Table 4 | core `e3_scale_invariance.objectives.privacy`, ext `B_constrrl_scale` | same |
| (c) contention 0 to 0.6, Lexi, Constr-WS, WS-tuned, 15 seeds | see EXT Table 6 | core `e7_misspecification.structural_contention` | same |

The regenerated `fig_results.pdf` is pixel-identical to the paper's figure with the
pinned packages.

### Sect. 6.1 (RQ1/RQ2)

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| One of 84 weightings meets SLO <= 8% and PII <= 0.1, none dominate Lexi | 0.012, 0.0 | core `e2_weight_sweep` |
| Lexi 1.42% SLO in the 6-seed sweep | 1.42 | core `e2_weight_sweep.lexi_strict` |
| Lexi 1.46%, 95% CI [1.31, 1.60], PII 0.05 | 1.458, [1.308, 1.600], 0.054 | core `e1_satisfaction.table.Lexi-strict` |
| Better on SLO than WS-tuned (paired p < 0.001), WS-tuned better on privacy | p = 0.0, 0.013 < 0.054 | core `e1_satisfaction.paired_tests` |
| Scale-robust sums at 2.3 to 3.3x Lexi's SLO | 2.26x to 3.34x | core `e1_satisfaction.table` (`rank_ws`, `norm_ws`, `tcheby`) |
| Only Lexi never inverts (0.0% vs 0.7 to 5.0%) | 0.0 vs 0.683 to 4.992 | core `e1_inversion.inversion` |

### Sect. 6.2 (RQ3)

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| Lexi PII invariant at 0.052 across a 300x swing | 0.052 at 0.1 to 30 | core `e3_scale_invariance.objectives.privacy` |
| WS-tuned PII 0.01 to 1.85, Tchebycheff 1.86 | 0.012, 1.848, 1.855 | same |
| Rank- and min-max-normalised sums near 3.5% SLO | 3.48, 3.42 | same (`rank_ws_slo`, `norm_ws_slo`) |
| Constr-WS 1.32% vs 1.46%, identical PII and carbon, cost within 0.02 | 1.317, 0.054, 0.984, cost gap 0.016 | core `e1_satisfaction.table` |
| Constr-WS inverts on 0.68% here and 14.3% under real Azure load, Lexi 0% in both | 0.683, 14.325, 0.0, 0.0 | core `e1_inversion`, real `inversion` |
| Constr-WS exposure 0.052 to 1.84 under 10x under-scaling | 0.052, 1.837 | ext `B_constrrl_scale.rows` |

### Sect. 6.3 (RQ4)

| Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- |
| Random menu moves Lexi from 1.44% to 43.6% (8 seeds) | 1.44, 43.56 | core `e5_candidate.sets` |
| 13.7%, 45.0%, 5.6% of curated candidates meet SLO, PII-clean, both | same | ext `A_menu.sets.curated` |
| Random menu shares 3.9%, 19.4%, 0.2% | same | ext `A_menu.sets.random` |
| Oracle 1.48% vs Lexi 1.44% (curated), 43.53% vs 43.56% (random) | same | ext `A_menu.sets` |
| Ordering on the random menu: 43.6% Lexi, 45.8% WS-tuned, 59.8% WS-default | 43.56, 45.77, 59.75 | core `e5_candidate.sets.random` |
| Ordering holds over SLO 220 to 340 ms | holds at 220, 260, 300, 340 | core `e6_generalization.slo_sweep` |
| Config B: Lexi 6.78% < Constr-WS 7.17% < WS-tuned 8.14% < WS-default 13.81% | same | core `e6_generalization.config_b` |
| Contention: Lexi 1.46% to 40.97% at c = 0.6, ordering kept except at the extreme | same | core `e7_misspecification.structural_contention` |
| PII exposure 0.05 to 1.49 under contention | 0.054, 1.491 | rob `rx1_contention.rows` (`additive_pii`) |
| SLO-hinge MAE 0.0286 over the whole menu | 0.0286 | ext `C_pred_acc.slo_hinge_mae_norm` |
| Lexi selects the expected-objective optimum on 99.7% of decisions | 99.73 | dec `oracle_agreement.curated.lexi_pct` |
| Under 0.1 ms per decision | 0.086 ms at 32 and 0.088 ms at 48 candidates (machine-dependent, INFO) | core `scalability.points` |

## Extended version

Tables render with `python3 make_tables.py extended`.

| Location | Claim in the paper | Artifact value | Source |
| :-- | :-- | :-- | :-- |
| Abstract, Sect. 1, 13, 15 | same headline numbers as the camera-ready | see above | core, ext |
| Sect. 3.2 | divisors 2 stages, 2.5 g, 5 cost units | same | sim `SCALES` |
| Sect. 3.3 | candidate #18 at (0.076, 0, 0.673, 0.888), 4.5 ms over the SLO | same, 264.54 ms predicted | dec `worked_decision.cand18_*` |
| Sect. 3.3 | #6 predicted 7 ms faster than #21, #21 was the previous placement | 6.8 ms, incumbent 21 | ext `panel`, dec `worked_decision.incumbent` |
| Sect. 3.3, 4.5 | tuned sum 0.030 vs 0.111, 0.021 vs 0.030 under 10x | same as CR Sect. 3 | dec `worked_decision` |
| Sect. 4.1 | time-of-day fraction 0.767 | 0.767 | ext `D_worked_example.request` |
| Sect. 4.3 | #24 and #33 admitted, true 268.2 and 264.8 ms | same | ext `panel` |
| Sect. 4.2 | model fitted on steps 0 to 903 (corrected from "1 to 903") | decision at step 904. **Resolved** | ext `D_worked_example.step` |
| Sect. 4.3 | predicted carbon of {2, 19, 20, 21}: 1.683, 1.190, 0.968, 0.784 g (corrected from the noise-free 1.687, 1.501, 0.969, 0.785) | same. **Resolved** | dec `worked_decision.level1_pred_carbon_g` |
| Sect. 4.3, 4.5 | #21 noise-free objectives (corrected from "realised"): 235.5 ms, PII 0, 0.785 g, $2.96. #6: 228.8 ms, 0.416 g, $1.08 | same. **Resolved** | ext `panel[].true_raw` |
| Sect. 8.1 | intensity with a per-node phase (corrected from "per-region") | phases differ within a region, for example 3.8, 3.5, 4.0 in region 2. **Resolved** | sim `build_nodes` (`carbon_phase`) |
| Table 1 | all eleven level-0 survivors | all match | ext `D_worked_example` |
| Table 3 | CR Table 3 rows plus Rank-WS, Norm-WS, Tchebycheff, SO-RL, static-greedy / binpack | all match | core `e1_satisfaction.table`, `e1_inversion` |
| Sect. 9.2 | Lex-LP byte-identical, agreement 1.00 over 24k decisions | 1.0, 24,000 | core `lexlp_agreement` |
| Sect. 9.2, 10, Table 7 | cold start: priced methods invert 22 to 37%, Lexi 0%, margin variant 22% vs WS-tuned 34% at 30 ms | 22.3 to 36.5, 0.0, 22.3, 33.6 | core `e11_conflict_stress.levels` |
| Sect. 9.3 | Constr-WS inversion 0.68%, CI [0.55, 0.83] | 0.683, [0.55, 0.833] | core `e1_inversion` |
| Table 4 | PII under privacy rescaling, Lexi, Constr-WS, WS-tuned, Tchebycheff | all 24 cells match | core `e3_scale_invariance`, ext `B_constrrl_scale` |
| Table 5 | menu anatomy, oracle and Lexi, gap 0.0 and 3e-5 | all match | ext `A_menu` |
| Sect. 9.4, 9.5 | optimum picked on 99.7% / 98.9% (Lexi), 99.1% / 87.6% (Constr-WS) | 99.73, 98.86, 99.06, 87.59 | dec `oracle_agreement` |
| Sect. 9.4 | SLO 220 ms: 58.9 / 58.9 / 73.8 / 88.0. 340 ms: 0% and WS-default PII 2 | 58.95, 58.95, 73.81, 87.95, 0, 2.0 | core `e6_generalization.slo_sweep` |
| Sect. 9.4 | config B PII: Lexi 0.00, WS-default 2.73 | 0.0, 2.725 | core `e6_generalization.config_b` |
| Table 6 | contention rows c = 0 to 0.6, Lexi PII, M/G/1 recovery 57/19/12/2% | all match | core `e7_misspecification`, rob `rx1_contention` |
| Sect. 9.5 | latency MAE 10.5 ms of 363.8 ms (2.9%), hinge 0.0286 (1.7 ms), carbon 0.053 g, cost $0.16 | 10.524, 363.8, 2.89, 0.0286, 1.714, 0.0527, 0.1608 | ext `C_pred_acc` |
| Sect. 9.5 | fleet carbon range roughly 0.3 to 2.6 g | 0.229 to 2.668 g over the menu | dec `menu_carbon_range` |
| Sect. 9.5 | DR study mean error 0.0246 | 0.0246 | rob `rx2_doubly_robust.rows[0]` |
| Sect. 9.6 | margin 4 ms: 0.16%, 8 ms: 0.01%, PII 0.05 / 0.11 / 0.28, p < 0.001 | 0.158, 0.008, 0.054 / 0.114 / 0.278, p = 0 | core `e10_margin` |
| Sect. 9.6 | contention 0.4: margin cuts 18.7% to 14.6% | 18.658, 14.6 | core `e10_margin.contention` |
| Sect. 5.3, 9.6 | churn 1.35 to 1.14, SLO 1.46 to 0.92% over slack 0 to 8 | 1.3455, 1.1378, 1.458, 0.917 | core `e4_stability` |
| Sect. 9.6 | slope 1.12e-4 ms/candidate, CI [1.05, 1.35]e-4, 0.082 to 0.096 ms (machine-dependent) | same in the committed file | core `scalability` |
| Sect. 8.2, 11 | regional carbon means 92.8, 167.5, 182.9 | same | real `provenance` |
| Table 8 | real-trace rows, all methods | all 41 cells match | real `table`, `inversion` |
| Sect. 11 | paired p = 0.15, priced scalarisers invert 16 to 17%, constrained 14% | 0.145, 16.3 to 16.8, 14.3 | real `paired_tests`, `inversion` |
| Table 9 caption, Sect. 12 | 15 seeds unless marked, dagger rows 2 to 5 seeds (corrected from "15 seeds, 95% CIs") | 15 seeds for RX1, RX2, RX4, RX6, RX7 and DX1 to DX7. Scaling 5 seeds, larger sweep 2 and 3 seeds. **Resolved** | rob `n_seeds`, `rx3_expanded_sweeps.n_seeds_sweep`, `n_seeds_baseline`, dep `n_seeds` |
| Table 9, Sect. 12 | M/G/1 57%, DR decision-safe (<= 0.5%), hinge rho invariant, k = 2 gives delta 3.7 ms and 0.23%, <= 0.75 ms at C = 2000, 220 weightings with 0% dominating, residency 2.7% to 0%, coverage 85.7% to 100% with SLO 1.46% to 1.30%, drift 37.2% vs 28.9%, planner 1.9% and 99%, PII 1.49 at c = 0.6, quantile agreement 100% | all match (timing as INFO) | rob `rx1` to `rx7`, dep `r2q1` to `r2q7` (keys of DX1 to DX7) |
| Sect. 12 | delta 0 / 4 / 8 ms: SLO 1.45 / 0.16 / 0.008%, PII 0.054 / 0.114 / 0.278, invariance lost only with privacy slack | 1.45, 0.158, 0.008, 0.0543, 0.1135, 0.278 | dep `r2q6_delta_slack` |

## Claims that are not numbers from a result file

Theorem 1, Propositions 1 to 3 and Assumption 1 are analytical and have no artifact
counterpart beyond the checks above (scale invariance of the strict choice, Lex-LP
agreement). Table 10 of the extended version is a qualitative comparison. Statements
about external systems (HuntMS, ClusterLess, Festina) are literature claims.
