# Export Report

## Overview
This workflow automates report generation by gathering metrics, generating charts, creating a PDF, and saving the output.

## Steps
1. Gather all fairness metrics from the current dataset.
2. Generate score charts and a summary explanation.
3. Create a PDF report with executive summary, metrics, and recommendations.
4. Save the PDF to `output/` and make it available for download.

## Automation Coverage
- Collects fairness metrics automatically from the detected protected groups.
- Builds an audit report with explanation and mitigation recommendations.
- Uses PDF generation for repeatable, shareable output.
- Saves the report to `output/bias_audit_report.pdf` or a similar path.

## Expected Output
- `bias_audit_report.pdf` in the `output/` folder.
- A PDF containing fairness scores, explanation, and mitigation guidance.

## Notes
The workflow is designed to support a consistent audit export process for demo and prototype use cases.