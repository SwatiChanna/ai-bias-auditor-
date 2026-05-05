# Fix Bias

## Overview
This workflow automates the one-click bias fixing pipeline for a loaded demo dataset.

## Steps
1. Detect bias in the loaded dataset using fairness metrics.
2. Recommend mitigation strategies based on the detected bias.
3. Apply AIF360 reweighing to rebalance the dataset.
4. Show before/after fairness scores for comparison.

## Automation Coverage
- Detects bias automatically from the selected protected attribute and label.
- Generates mitigation recommendations for data, model, and output level.
- Applies reweighing preprocessing automatically.
- Displays before and after metrics in a single flow.

## Expected Output
- `before_scores`: Original disparate impact and statistical parity difference.
- `after_scores`: Reweighed disparate impact and statistical parity difference.
- `instance_weights`: Reweighted sample weights added to the dataset.

## Notes
This workflow is intended to reduce manual bias-fixing steps for demo datasets and validation flows.