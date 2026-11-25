# Interview guide

Questions an interviewer is likely to ask about FraudShield AI, with short answers and where
to find the code. Read the code behind each answer, so you can explain it in your own words.

## The problem

**What problem does this solve?**
Fraud teams can review only a tiny share of transactions. FraudShield ranks every
transaction by fraud risk, groups them into tiers with an action, and explains each score,
so analysts spend their time on the riskiest ones. It supports decisions; it does not make
them.

**Why is fraud detection hard for machine learning?**
Fraud is rare (0.39% here), so a model can be 99.6% accurate by approving everything. The
two kinds of error also cost different things: a missed fraud loses money, a false alarm
annoys a customer and costs analyst time.

## Data

**What data did you use and why?**
The Sparkov simulated card dataset (1.85 M transactions). Its features are readable
(amount, category, time, location, age), which makes EDA and explanations meaningful. The
popular ULB dataset has anonymised PCA features (V1–V28), which nobody can explain to a
business user. → `docs/dataset.md`

**How do you handle bad data?**
Every value is read as text first, then converted, so failures can be counted instead of
silently becoming blanks. Each problem type has a fixed, documented action (remove row,
clear value, or warn), and the app shows a quality report. → `src/fraudshield/data/validation.py`

**Why keep card numbers as text?**
A 16-digit card number stored as a float loses its last digits (floats hold about 15
significant digits). → `validation.py::_convert`

**How do you handle large files on a small laptop?**
Read in 100,000-row chunks, only the needed columns, repeated text stored as pandas
`category`, and tables read one page at a time from SQLite. 1.3 M rows validate in about
16 s with under 1 GB of memory.

## Machine learning

**How did you split the data and why?**
By time: train Jan 2019 – Mar 2020, validate Mar – Jun 2020, test Jun – Dec 2020. In real
use a model always predicts the future; a random split would let it learn from transactions
that happen after the ones it is tested on. → `models/training.py::time_split`

**How did you prevent data leakage?**
Everything learned from data (median imputation, scaling, one-hot categories, each card's
typical spending) is a step inside one scikit-learn `Pipeline`, fitted on training rows
only. Missing values are deliberately not filled in during data cleaning, because the median
must come from training data only. Tests check that the card profile never changes when test
data passes through it. → `features/engineering.py::CardProfile`, `tests/test_models.py`

**Why a "card profile" feature instead of "transactions in the last 24 hours"?**
A velocity feature needs every transaction of a card. A sample or uploaded extract contains
only some of them, so the same feature would mean something different at training and at
scoring time (train/serve skew). The card profile is learned once from complete training
data and then looked up.

**How did you deal with class imbalance?**
Balanced class weights (no extra library needed), then calibration, then a decision
threshold chosen on validation data. I didn't use SMOTE: class weights did the job without
creating synthetic rows.

**What is calibration and why does it matter?**
With class weights, a model's scores are no longer real probabilities. Isotonic calibration,
fitted on the validation period, maps scores to probabilities. Proof it worked: transactions
in the Critical tier (85%+) were 94% fraud on the test period.

**Why PR-AUC and not accuracy or ROC-AUC?**
Accuracy is fooled by rare fraud. ROC-AUC looks good for most models because even many
false alarms are a small share of 550,000 legitimate transactions. PR-AUC focuses on the
fraud class; its random baseline is the fraud rate (0.004), so 0.823 is about 200× better.

**Why did Random Forest win so clearly?**
Fraud here depends on combinations (a large amount *and* a risky category *and* late at
night). Logistic Regression can only add effects up in straight lines (PR-AUC 0.25); a
single shallow tree is too simple (0.34); many trees together capture the interactions
(0.82).

**What is Isolation Forest doing in a supervised project?**
It is an unsupervised baseline: it finds unusual transactions without fraud labels. It was
26× better than random but far behind the supervised models, which shows the labels carry a
lot of information. It is compared at the same analyst workload, because it has no
probabilities.

**How did you choose the model and threshold without cheating?**
Both on validation data. The test set is used once, at the end, and never for a decision.

## Explainability and responsibility

**How do you explain a single prediction?**
For each feature group, replace the transaction's value with values from 200 typical
transactions, re-score, and measure how much the probability changes. It is model-agnostic
and needs no extra library. → `models/explain.py`

**Which features did you exclude and why?**
Gender (protected attribute), job and merchant (hundreds of values, overfitting and
near-personal data), city/state (location already covered by distance and city size),
identifiers. Age is used; before real deployment outcomes should be checked across age groups.

## Engineering

**How is the project structured?**
All logic is in the `fraudshield` package; Streamlit pages only display results. That
makes the logic testable without a browser. → architecture diagram in the README

**Why Streamlit and not Django?**
A single-user analytics tool needs interactive charts and file upload, not user accounts or
a custom frontend. Streamlit is one lightweight process; Django would add a lot of code
without showing more data science.

**Why SQLite?**
Built into Python, no server, survives restarts, and lets tables show 50 rows at a time
instead of loading a million into memory.

**How did you test it?**
About 190 automated tests at 98% line coverage: unit tests against hand-calculated values,
leakage tests, an end-to-end test from CSV to PDF, Streamlit tests that click through the real
pages on a temporary database, and security tests (SQL injection text, tampered model file).
→ `docs/testing.md`

**Tell me about a bug you found.**
Explanations were wrong whenever Logistic Regression was the selected model. scikit-learn
calibrates models that have a `decision_function` on that score, but my explanation code
passed `predict_proba`. It didn't affect the Random Forest, so ordinary testing missed it; an
end-to-end test caught it. I fixed it and added a test that checks every model type.

**What security measures are there?**
Parameterised SQL (user text never becomes SQL), escaped HTML and Markdown, masked card
numbers, personal columns never loaded, and a SHA-256 fingerprint check before loading the
model file (joblib files can run code).

## Limitations and next steps

**What would you do with more time?**
Analyst feedback and periodic retraining, drift monitoring, gradient boosting with SHAP, a
real-time scoring API, and a fairness review across age groups.

**What are the main limitations?**
Simulated data, a static model, one in four fraud cases still missed at the chosen
threshold, and single-user local deployment.
