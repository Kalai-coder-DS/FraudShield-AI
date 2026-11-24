# Dataset guide

## Source

**Credit Card Transactions Fraud Detection Dataset** (Kaggle user kartik2112),
generated with the Sparkov data-generation tool. It simulates card transactions of
about 1,000 customers with 800 merchants from 1 Jan 2019 to 31 Dec 2020.

| File | Rows | Period | Used for |
|---|---|---|---|
| `fraudTrain.csv` | 1,296,675 | Jan 2019 – Jun 2020 | Model training (Phase 4) |
| `fraudTest.csv` | ~0.56 million | Jun 2020 – Dec 2020 | Out-of-sample testing and the bundled sample |

The data is **synthetic**: no real person's transactions are included. Check the
licence on the Kaggle page before redistributing the full files. Only a small sample
is committed to this repository; the full files are git-ignored.

## Setup

1. Download the dataset from Kaggle and unzip it.
2. Copy `fraudTrain.csv` and `fraudTest.csv` into `data/raw/`.
3. Build the sample used by the app:

   ```bash
   python scripts/make_sample.py            # ~50,000 rows from fraudTest.csv
   python scripts/make_sample.py --rows 20000   # smaller, for low-memory laptops
   ```

The sample is taken from the **test** file on purpose: the model is trained on the
training file, so transactions in the sample were never seen during training. It is
stratified (the same share of fraud and legitimate rows is kept from every chunk),
so its fraud rate matches the full file.

## Canonical schema

Every dataset is mapped to these columns before use (`src/fraudshield/data/schema.py`).
Sparkov columns are matched automatically.

| Column | Required | Sparkov column | Meaning |
|---|---|---|---|
| transaction_id | ✓ | trans_num | Unique transaction ID |
| timestamp | ✓ | trans_date_trans_time | Date and time |
| amount | ✓ | amt | Amount (must be ≥ 0) |
| card_id | ✓ | cc_num | Card that paid (kept as text; masked in the UI) |
| category | ✓ | category | Spending category |
| is_fraud | | is_fraud | 1 = fraud, 0 = legitimate |
| merchant, gender, dob, job, city, state, city_pop | | same names | Customer and merchant details |
| lat, long, merch_lat, merch_long | | same names | Customer and merchant location |

Not loaded: `first`, `last`, `street` (personal data, never needed), `zip`,
`unix_time` (duplicates the timestamp) and the unnamed index column.

## Validation rules

| Problem | Severity | Action |
|---|---|---|
| Required value missing or unreadable | Error | Row removed |
| Negative amount | Error | Row removed |
| Transaction ID reused with different details | Error | Later rows removed |
| Exact duplicate row | Warning | Copies removed |
| Optional value unreadable or out of range (e.g. latitude 123) | Warning | Value cleared |
| Age below 0 or above 120 at transaction time | Warning | Date of birth cleared |
| Optional value missing | Warning | Kept as missing |
| Amount of 0, date in the future | Warning | Kept |

**Quality score** = usable rows ÷ analysed rows × 100. A row is usable when it is not
a duplicate and all required fields are valid.

Missing optional values are *not* filled in at this stage. Filling them uses
statistics (such as a median) that must be learned from training data only;
doing it here, on the whole dataset, would leak test information into the model.

## Performance

Measured on a 1.3-million-row file in the Sparkov format (242 MB), on a 2-core
cloud machine:

| Step | Result |
|---|---|
| Read, validate and clean | 16 s |
| Memory of the cleaned table | 175 MB |
| Peak memory of the process | 651 MB |
| Save to SQLite | 8 s |
| Read one page of 50 rows | 10 ms |

How memory is kept low: the file is read 100,000 rows at a time
(`CSV_CHUNK_SIZE` in `config.py`), only mapped columns are read, repeated text such
as category and card number uses pandas' `category` type, and tables in the app read
one page at a time from SQLite. On a laptop with 4 GB RAM, load 25–50% of the full
training file using the "Share of rows to load" option, or use the sample.
