# Exploratory Data Analysis (Module 3)

The EDA page analyses the **active dataset**. All numbers come from
`src/fraudshield/eda/analysis.py`; charts in `charts.py` only draw them.

## What is analysed

| Tab | Content |
|---|---|
| Summary | Key findings in plain language, fraud vs. legitimate balance, daily activity |
| Amounts | Amount distribution per class (log scale), statistics table, outliers (1.5 × IQR), fraud rate by amount range |
| Time patterns | Fraud rate by hour and weekday, transaction volume by hour |
| Categories & customers | Fraud rate by category, age group and gender |
| Relationships | Spearman correlation heatmap, amount box plots per category and class |

Datasets without a fraud label show only the general patterns (volumes, amounts,
correlations).

## Methods and why they were chosen

- **Fraud rates on labelled rows only.** Rows with an unknown label are excluded
  from every rate.
- **95% confidence intervals (Wilson score).** A rate measured on 20 transactions is
  far less certain than one measured on 20,000. The intervals are drawn as error
  bars. Groups with fewer than 30 transactions are drawn in a muted colour.
- **Findings only for clear differences.** A category or hour is called out only if
  it has ≥ 30 transactions and the lower end of its interval is above the overall
  fraud rate. Random noise is therefore not reported as a pattern.
- **Shares within each class** for the amount distribution, so the rare fraud class
  is visible next to the large legitimate class.
- **Log scale for amounts**, because amounts are very skewed.
- **Spearman correlation** instead of Pearson: it works on ranks, so a few huge
  amounts do not dominate, and it detects any consistently rising or falling
  relationship, not just straight lines.
- **Outliers are kept.** In fraud detection an unusual amount is often the signal.
- **Gender is shown for analysis only.** Using protected attributes in a model needs
  a fairness review (Phase 4 decides and documents this).

## Row features (shared with Phase 4)

`src/fraudshield/features/basic.py` adds hour, weekday, date, log amount, customer
age at transaction time and customer–merchant distance (haversine). Each value uses
only its own row, so these features cannot leak information between training and
test data.

## Performance

Summaries are computed once per dataset and cached (`st.cache_data`); only the small
summary tables (about 22 KB) are kept, not the transactions. On 1.3 million rows all
summaries take about 3 seconds the first time.
