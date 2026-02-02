# DeepONet Model Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Test Samples | 10 |
| Total Points | 8,318,651 |
| Avg Inference Time | 288.70 ms |
| Speedup vs CFD | ~3x faster (estimated) |

## Quantitative Metrics

| Variable | R² Score | RMSE | MAPE (%) |
|----------|----------|------|----------|
| pressure | 0.7851 | 3.7070e+05 | 471126476800.00 |
| density | 0.7531 | 1.1396e+00 | 71.98 |
| velocity-magnitude | 0.6769 | 2.4278e+02 | 23.26 |
| mach-number | 0.7884 | 7.1543e-01 | 36.39 |
| temperature | 0.7356 | 1.6728e+02 | 21.88 |

## Generated Visualizations

### Statistical Graphs
- `metrics/scatter_all_fields.png` - Predicted vs Ground Truth scatter plots
- `metrics/error_histograms.png` - Error distribution histograms
- `metrics/metrics_comparison.png` - Variable-wise metric comparison

### Flow Field Comparisons
Contour plots showing CFD (left) vs DeepONet (right) with error maps:
- `contours/sample_*_pressure_comparison.png`
- `contours/sample_*_mach_number_comparison.png`  
- `contours/sample_*_density_comparison.png`

### Training History
- `training_history.png` - Loss curves and learning rate schedule

## Notes
- R² values close to 1.0 indicate excellent prediction accuracy
- MAPE shows percentage deviation from ground truth
- Inference time measured on cuda
