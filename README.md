# AI Bias Auditor

A demo tool for detecting, analyzing, and mitigating AI bias across gender, caste, language, and region dimensions.

## 2-Minute Demo Walkthrough

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. In the app Home section, choose one of the demo datasets:
   - `Indian Loan Audit (Caste)`
   - `Regional Hiring` 
   - `Language Bias`
   - `Gender Income Gap`
4. Open `Counterfactual Testing`, select a dimension, and click `Run Counterfactual Test`.
   - The app swaps the protected attribute and compares original vs counterfactual fairness scores.
5. Open `Fairness Dashboard` to see the current bias score.
   - Apply AIF360 Reweighing to compare before and after mitigation.
6. Export the audit by downloading the PDF report in the `Fairness Dashboard` section.

## Workflow Files

- `.agents/workflows/generate-bias-tests.md`
- `.agents/workflows/fix-bias.md`
- `.agents/workflows/export-report.md`

## Notes
The app produces reusable metrics and PDF audit reports, and the new workflow files document repeatable steps for bias testing, mitigation, and reporting.