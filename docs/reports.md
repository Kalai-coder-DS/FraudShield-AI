# Executive Dashboard and Reports (Modules 1 and 8)

Both read one `Summary` object (`src/fraudshield/reports/summary.py`), built from the
active dataset, its stored scores and the model metadata. The dashboard, the CSV and the
PDF therefore always show the same numbers.

## Executive Dashboard

For a non-technical reader:

- a one-sentence headline ("N transactions need an analyst's attention today"),
- transactions analysed, total value, flagged for review, value flagged,
- High + Critical count and value, known fraud and how much of it was flagged (if labelled),
- model quality on unseen test data (PR-AUC against the random baseline),
- risk tier chart, categories over-represented among High/Critical (lift), weekly trend,
- findings written from the numbers, and links to the next pages.

If the dataset overlaps the model's training period, a warning says the results are
optimistic.

## Reports

| Download | Content |
|---|---|
| CSV | Scored transactions (all, flagged only, or High + Critical), riskiest first. Card numbers masked to the last 4 digits. UTF-8 with BOM so Excel opens it correctly. |
| PDF (2–3 pages) | Key figures, findings, risk tier table and chart, weekly activity, risky categories, model performance and comparison, disclaimer, page numbers. |

The PDF is built with ReportLab and Matplotlib (static charts suit print). The project
originally planned fpdf2; ReportLab was chosen because it could be tested end to end in the
development environment, including reading the generated PDF back in the tests to check
that its numbers match the data.

## Presentation choices

- Probabilities are shown as "> 99.9%" / "< 0.1%" at the extremes: a model is never certain.
- Activity is shown per week (daily fraud counts are small and noisy); partly covered first
  and last weeks are left out so they don't look like sudden drops; axes start at zero.
