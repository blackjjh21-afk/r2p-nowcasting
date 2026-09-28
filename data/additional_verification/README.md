# Additional verification data

Aggregate results for rainfall-episode peak verification and the exPreCast
CNN 3 × 3 versus 5 × 5 sensitivity experiment. These tables correspond to
Fig. 6 and Supplementary Figs. S2–S3 in the Journal of Hydrology manuscript.

| Files | Contents |
|---|---|
| `episode_route_metrics.csv`, `episode_member_metrics.csv` | Peak-amount and maximum-timing errors, pooled and by year |
| `episode_counts.csv` | Observation-based episode selection and timing eligibility counts |
| `episode_duration_statistics.csv` | Duration summaries for the observation, amount and timing samples |
| `episode_duration_histogram.csv` | Episode counts in 10-minute duration bins, each episode counted once |
| `patch_metrics_mean_of_members.csv`, `patch_metrics_per_member.csv` | Skill for exPreCast + CNN3, exPreCast + CNN5 and Direct R2P |
| `patch_paired_CSI_intervals.csv` | Paired issuance-date-bootstrap CSI differences and 95% intervals |
| `patch_epoch_selection.csv` | All 50 OOF epoch scores and the selected epoch |

Episodes are observation-defined contiguous RN60 ≥ 5 mm intervals containing
an observed maximum ≥ 20 mm. Duration is the number of 10-minute slots times
10 minutes, not the duration of uninterrupted rainfall. Amount errors use
1,437 episodes. Maximum-timing errors use the common 1,391-episode subset
with non-constant forecasts in every route, member and lead. Errors are
computed for each member before averaging across the three route members.

Histogram bins are left-closed and right-open, for example [50, 60) and
[60, 70) minutes. The 1-hour tick is their shared boundary.

`manifest.json` gives the schema and checksum for each table. Station-resolved
episode values are provided separately in Supplementary Data 1; the software
bundle contains only the aggregate tables above.

To reproduce the three final figures from these tables, run from the repository root:

```bash
python -m paper_outputs.render_additional_verification \
  --data-root data/additional_verification \
  --output-dir outputs/additional_verification
```

The default outputs are `Figure_6`, `Figure_S2` and `Figure_S3` PNG/PDF pairs.
The CSI-difference plot is omitted from the submission. Its numerical
differences and confidence intervals remain in `patch_paired_CSI_intervals.csv`;
`--include-contrasts` optionally renders that audit plot.

For recomputation from authorized observations and predictions, see
[rainfall episodes](../../src/evaluation/RAINFALL_EPISODES.md) and
[CNN patch sensitivity](../../src/cnn_readout/PATCH_SENSITIVITY.md).
