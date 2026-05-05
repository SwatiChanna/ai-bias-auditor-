# AGENTS.md

Standing instructions for all agents and developers working on the AI Bias & Fairness Auditor.

## Project Context

This is a 24-hour hackathon project for building an AI system that detects bias across gender, caste, language, and region using counterfactual testing.

Tech stack:

- Python
- Streamlit
- AIF360
- Fairlearn
- SHAP
- Plotly
- pytest
- python-dotenv

Team workflow:

- Two developers may work in parallel using Google Antigravity.
- Keep changes small, modular, and easy to merge.
- Do not rewrite unrelated files.
- Prefer clear, demo-ready behavior over over-engineered abstractions.

## Code Style Rules

- Follow PEP 8 for all Python code.
- Type hints are required for every function signature.
- Use Google-style docstrings for all public functions, classes, and modules.
- Keep functions focused and testable.
- Avoid hidden global state.
- Use clear names that describe fairness, bias, protected attributes, and counterfactual behavior.
- Keep imports grouped in this order: standard library, third-party packages, local modules.
- Do not leave unused imports, dead code, commented-out experiments, or notebook-only snippets.

Example docstring style:

```python
def calculate_bias_score(y_true: list[int], y_pred: list[int], group_labels: list[str]) -> float:
    """Calculate a bias score for model predictions across protected groups.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        group_labels: Protected group labels aligned with each prediction.

    Returns:
        A normalized bias score where higher values indicate greater disparity.
    """
```

## Testing Rules

- Use pytest for all tests.
- Every function in `src/` must have at least one unit test.
- Minimum coverage is 80%.
- Tests must cover normal cases, edge cases, and invalid inputs where relevant.
- Bias scoring tests must include at least one fair case and one biased case.
- Counterfactual tests must verify that swaps preserve unrelated fields.
- Dashboard logic should be separated from Streamlit rendering so it can be tested.
- Do not skip tests unless the reason is documented and temporary.

## Architecture Rules

- `main.py` is the entry point only.
- Keep Streamlit rendering thin and route all feature logic into `src/` modules.
- Do not place fairness scoring, mitigation logic, SHAP logic, or counterfactual generation directly in the entry point.
- Use this module ownership:
  - `src/bias_engine.py`: fairness metrics, group disparity scoring, counterfactual evaluation.
  - `src/explanations.py`: SHAP explanations and plain-English summaries.
  - `src/mitigation.py`: mitigation recommendations and intervention ranking.
  - `src/counterfactual_templates.py`: gender, caste, language, and region swap templates.
- Keep data loading isolated from analysis logic.
- Use typed data structures where they improve clarity.
- Avoid circular imports.
- Prefer pure functions for scoring, transformations, and recommendations.

## Security Rules

- Never hardcode API keys, secrets, passwords, tokens, or private URLs.
- Use `python-dotenv` for local configuration.
- Load environment variables from `.env` only through a centralized configuration helper.
- Do not commit `.env` files.
- Validate uploaded files before processing.
- Do not execute user-uploaded content.
- Avoid logging sensitive input rows or personally identifiable information.
- Demo datasets must not contain real private user data.

## Indian Context Rules

- The product must explicitly support caste bias analysis using the IndiCASA dataset or compatible caste-labelled datasets.
- The product must support Indic language code-mixing, including examples where English is mixed with Hindi or other Indian languages.
- Do not treat Indian fairness dimensions as cosmetic labels. Gender, caste, language, and region must each have actual test logic.
- Use respectful, neutral terminology for caste, region, language, and identity groups.
- Avoid generating derogatory or harmful identity text.
- When showing examples, prefer neutral professional, educational, financial, or public-service scenarios.
- Treat caste as a protected attribute.
- Treat language and region as potential proxies for socioeconomic and cultural bias.

## Counterfactual Testing Rules

Counterfactual testing is a required core feature. Every supported model or scoring path must be able to compare an original input with identity-swapped variants.

Required dimensions:

- Gender swap
- Caste swap
- Language swap
- Region swap

Counterfactual generation must:

- Preserve all non-protected attributes unless a specific template requires a related contextual phrase to change.
- Record which fields were changed.
- Record the original value and counterfactual value.
- Keep a stable case ID linking original and counterfactual examples.
- Return outputs in a structure that can be scored, displayed, and tested.
- Avoid changing labels or ground truth during generation.
- Support batch generation for demo datasets.

## Gender Counterfactual Instructions

Gender swaps must support:

- Male to female terms.
- Female to male terms.
- Neutral or unknown gender terms where applicable.
- Pronoun swaps such as he/she, him/her, his/her.
- Title swaps such as Mr./Ms. where relevant.
- Relationship terms such as father/mother, son/daughter, husband/wife where present.

Rules:

- Preserve job role, education, income, region, language, and caste unless explicitly testing an interaction effect.
- Do not introduce stereotypes.
- Keep names optional. If names are swapped, use neutral placeholder names or a documented name mapping.

## Caste Counterfactual Instructions

Caste swaps must support:

- General category.
- OBC.
- SC.
- ST.
- Other categories present in the dataset.
- IndiCASA-compatible caste labels.

Rules:

- Treat caste as a protected attribute.
- Preserve education, income, region, language, gender, and occupation unless explicitly testing an interaction effect.
- Never generate insulting, stigmatizing, or hierarchical wording.
- Prefer structured attributes such as `caste_group` over free-text caste descriptions.
- For text examples, use neutral phrasing such as "candidate from SC category" only when needed for the test case.

## Language Counterfactual Instructions

Language swaps must support:

- English.
- Hindi.
- Hinglish or Hindi-English code-mixing.
- Other Indic languages represented in the dataset or demo cases.
- Formal and informal variants where relevant.

Rules:

- Preserve intent, qualifications, sentiment, and factual content.
- Change only language style or language identity marker.
- Include code-mixed examples such as English sentences containing Hindi words or Hindi written in Latin script.
- Do not degrade grammar or professionalism in a way that confounds the test.
- Track whether the test changes script, language, or code-mixing level.

## Region Counterfactual Instructions

Region swaps must support:

- North India.
- South India.
- East India.
- West India.
- Northeast India.
- State-level or city-level variants where available.

Rules:

- Preserve qualifications, income, gender, caste, and language unless explicitly testing interactions.
- Avoid stereotypes about regions.
- Prefer structured fields such as `region`, `state`, or `city`.
- If text mentions a place, swap it with a comparable place rather than changing socioeconomic status.
- Track whether the swap is broad region, state, city, or dialect-related.

## Bias Scoring Rules

- Bias scores must map to a clear traffic-light status:
  - Red: high bias or unacceptable disparity.
  - Yellow: moderate bias or warning-level disparity.
  - Green: low bias or acceptable disparity.
- Thresholds must be documented in code and tests.
- Use Fairlearn and AIF360 metrics where appropriate.
- Counterfactual score differences must be shown separately from group aggregate metrics.
- Always expose the protected attribute used for each score.
- Include plain-English summaries explaining what changed and why it matters.

## Dashboard Rules

- The Streamlit dashboard must support a one-click demo dataset flow.
- Demo dataset choices must include:
  - Income prediction gender bias demo.
  - IndiCASA caste bias demo.
  - Language and region bias demo.
- Bias results must use traffic-light colors:
  - Red for high risk.
  - Yellow for medium risk.
  - Green for low risk.
- Use Plotly for charts.
- Show counterfactual examples side by side with original examples.
- Show SHAP explanations in plain English alongside visual explanations.
- Keep the demo path reliable even if optional user uploads fail.

## Collaboration Rules

- Before editing a shared file, understand its current purpose.
- Keep commits or change groups focused by feature.
- When parallel work may conflict, prefer adding new functions over rewriting shared logic.
- Add tests with each feature.
- Document assumptions in docstrings or README notes, not scattered comments.
- If a hackathon tradeoff is made, mark it clearly with `TODO(hackathon):` and explain the follow-up.
