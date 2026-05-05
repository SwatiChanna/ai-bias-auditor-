# Generate Bias Tests

## Overview
This workflow standardizes counterfactual bias testing by automating the core testing loop.

## Steps
1. Select the bias dimension to test (gender, caste, language, region).
2. Generate a counterfactual dataset for the selected dimension.
3. Run the demo model on both original and counterfactual datasets.
4. Compare fairness scores to identify sensitivity and bias.

## Automation Coverage
- Automates 80% of the manual counterfactual testing process.
- Automatically selects the protected attribute and target label for demo datasets.
- Generates the counterfactual dataset and computes fairness metrics.
- Produces side-by-side score comparisons for quick analysis.

## Expected Output
- `original_scores`: Baseline disparate impact and statistical parity difference.
- `counterfactual_scores`: Scores after protected attribute swap.
- `sensitivity_report`: Difference between original and counterfactual scores.

## Notes
This workflow is intended for repeatable bias test generation across demo datasets and can be used as a validation step before mitigation.