# SmartDesk AI – Intelligent Help Desk Management System

A complete, locally running help desk web application built with **Python, Streamlit, Scikit-learn, TF-IDF, Multinomial Naive Bayes, Pandas and SQLite**.

Users submit complaints through a web form. A genuinely trained Machine Learning model predicts the complaint category and shows its confidence. The system also assigns an explainable rule-based priority, suggests a first-line response, saves the ticket in SQLite, and lets an administrator manage every ticket from a dashboard.

> **Academic mini-project** for Third Year Engineering – Fundamentals of AI Lab.
> The dataset is a small demonstration dataset created for learning. It is not production data.

---

## 1. Project Introduction

Help desks receive many complaints every day: network failures, login problems, broken hardware, software errors and general questions. Reading and sorting each complaint manually takes time and can lead to inconsistent priorities and slower responses.

SmartDesk AI automates the **first pass** of help desk work:

1. The user submits a complaint.
2. The trained ML model predicts the category (for example, `Network Issue`).
3. The system displays the prediction confidence.
4. A rule-based module assigns `High`, `Medium` or `Low` priority.
5. A predefined suggested response is generated from the category.
6. The complete ticket is stored in SQLite.
7. The administrator views live statistics and can search, filter and update statuses.

Everything runs on the local machine. No cloud service, paid API or external database is required.

---

## 2. Problem Statement

In a college or small organisation, complaints are often collected through email, chat or paper. Support staff must manually:

- read every complaint,
- decide which category it belongs to,
- decide how urgent it is,
- write an initial response,
- and track its status.

This manual process is **slow, repetitive and inconsistent**. A Machine Learning classifier can learn word patterns associated with complaint categories and perform the first sorting step automatically, allowing staff to focus on resolution.

---

## 3. Objectives

- Build a working complaint submission form with proper validation.
- Train a real Scikit-learn classification model using labeled complaint data.
- Use TF-IDF for text vectorization and Multinomial Naive Bayes for classification.
- Display the predicted category **and the actual prediction confidence**.
- Detect priority using simple, explainable rules (not Machine Learning).
- Generate a category-based suggested response.
- Store all tickets permanently in SQLite using parameterized queries.
- Provide a live admin dashboard based only on real database records.
- Allow administrators to search, filter, inspect and update tickets.
- Verify the system with automated Pytest tests.
- Keep the code simple enough for a Third Year Engineering student to explain confidently.

---

## 4. Features

### User complaint submission
- Full name, email, subject and description fields.
- Required-field validation.
- Email format validation.
- Minimum description length validation.
- Unique readable ticket ID such as `TKT-000001`.
- Success message with ticket ID.
- Prediction category, confidence, priority and suggested response shown immediately.

### AI classification
- Real TF-IDF + Multinomial Naive Bayes pipeline.
- 100 labeled sample complaints across 5 categories.
- Stratified train/test split with a fixed random state.
- Model and vectorizer saved with Joblib.
- Graceful handling of unknown or low-confidence complaints.
- Prediction confidence displayed as a percentage.

### Priority detection
- Rule-based and explainable.
- Priority levels: `High`, `Medium`, `Low`.
- Shows which keyword/phrase triggered the priority.
- Clearly labelled as rule-based, **not** Machine Learning.

### Suggested response
- Category-specific predefined first-line response.
- Stored with the ticket in SQLite.

### Admin dashboard
- Total, Pending, In Progress and Resolved ticket counts.
- Charts for tickets by category, priority and status.
- Recent tickets table.
- All values come from SQLite queries, never hardcoded.

### Ticket management
- Search by ticket ID, subject, description or name.
- Filter by category, priority and status.
- View full ticket details.
- View AI category and confidence.
- View the suggested response.
- Update status to `Pending`, `In Progress` or `Resolved`.

### Evaluation transparency
- Accuracy, precision, recall and F1-score are calculated at training time.
- Full classification report and confusion matrix.
- Actual metrics are shown on the **About Project** page.
- Limitations of the small dataset are stated openly.

---

## 5. Technology Stack

| Purpose | Technology |
|---|---|
| Programming language | Python 3.11+ (tested on Python 3.12) |
| Web application | Streamlit |
| Machine Learning | Scikit-learn |
| Text vectorization | TF-IDF (`TfidfVectorizer`) |
| Classification algorithm | Multinomial Naive Bayes (`MultinomialNB`) |
| Database | SQLite (`sqlite3`, built into Python) |
| Data processing | Pandas |
| Model persistence | Joblib |
| Testing | Pytest |

No React, Node.js, Docker, cloud service, paid API or external database is used.

---

## 6. AI Methodology (Simple Explanation)

### Step 1 – Training data

`data/complaints.csv` contains 100 labeled complaint texts. Each row has:

- `text` – the complaint wording,
- `category` – one of five labels: `Network Issue`, `Account Issue`, `Hardware Issue`, `Software Issue`, `General Inquiry`.

### Step 2 – Train/test split

The dataset is split into **80% training data** and **20% testing data** using:

```python
train_test_split(..., test_size=0.20, random_state=42, stratify=target)
```

`stratify` keeps the category proportions similar in both sets. The test set is kept separate so evaluation is honest.

**No data leakage:** the TF-IDF vectorizer is inside the Scikit-learn pipeline. It is fitted only on the training text, never on the test text.

### Step 3 – TF-IDF vectorization

TF-IDF means **Term Frequency – Inverse Document Frequency**.

- **Term Frequency (TF):** how often a word appears in one complaint.
- **Inverse Document Frequency (IDF):** how rare that word is across all complaints. Very common words such as "the" receive a low weight.

A word gets a high TF-IDF score when it appears often in a complaint but rarely in other complaints. This makes words like "router", "password" or "printer" more informative than common English words. The vectorizer also learns **word pairs (bigrams)** such as "not printing" or "cannot login", which capture useful meaning.

The output is a numerical vector for every complaint, which the classifier can use.

### Step 4 – Multinomial Naive Bayes

Multinomial Naive Bayes is a probability-based classifier commonly used for text.

- During training it estimates how frequently each word/term occurs in each category.
- For a new complaint it calculates the probability of each category using those learned term frequencies.
- It assumes that words are conditionally independent given the category ("naive" assumption). This assumption is not perfectly true in language, but the algorithm still works well for text classification and is very fast.
- The category with the highest probability becomes the prediction.
- That highest probability is shown as the **confidence**.

The Scikit-learn pipeline is:

```text
Complaint text
     │
     ▼
TfidfVectorizer (words + word pairs, English stop words removed)
     │
     ▼
MultinomialNB (alpha = 0.3 smoothing)
     │
     ▼
Predicted category + probability for each category
```

### Step 5 – Handling uncertainty

- If the model recognizes none of the words from its training vocabulary, the ticket is routed to `General Inquiry` for manual review.
- If the highest confidence is below **30%**, the ticket is also routed to `General Inquiry` for manual review.
- The UI explains why this happened instead of pretending the prediction is reliable.

### Step 6 – Evaluation

The training script evaluates only on the held-out test set and reports:

- **Accuracy** – percentage of test complaints classified correctly.
- **Precision (macro)** – when the model predicts a category, how often it is right, averaged equally across categories.
- **Recall (macro)** – how many actual complaints of each category were found, averaged equally across categories.
- **F1-score (macro)** – harmonic mean of precision and recall.
- **Classification report** – per-category precision, recall, F1 and support.
- **Confusion matrix** – which categories were confused with which.

These values are **calculated, not hardcoded**. They are saved to `models/metrics.json` and displayed on the **About Project** page.

### Priority is not Machine Learning

Priority detection is a separate **rule-based** module (`services/priority.py`). It searches for transparent phrases:

| Priority | Example triggers |
|---|---|
| High | "system down", "no internet", "cannot access", "all users", "locked out", "critical" |
| Medium | "error", "repeatedly", "slow", "installation", "crash", "not responding" |
| Low | "how do I", "feature request", "suggestion", "guide", "pricing" |

If no rule matches, priority defaults to `Medium` for manual review. This is simpler, explainable and appropriate for an academic project. The project does **not** claim that priority is ML-based.

---

## 7. System Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit Web Interface                  │
│  Dashboard │ Submit Complaint │ Manage Tickets │ About       │
└───────────────┬───────────────────────┬──────────────────────┘
                │                       │
                ▼                       ▼
     ┌────────────────────┐    ┌──────────────────────┐
     │ services/          │    │ database/db.py       │
     │ validation.py      │    │ - SQLite connection  │
     │ classification.py  │    │ - CRUD operations    │
     │ priority.py        │    │ - statistics queries │
     │ response.py        │    └──────────┬───────────┘
     └─────────┬──────────┘               │
               │                          ▼
               ▼                 ┌────────────────────┐
     ┌────────────────────┐      │ SQLite             │
     │ ml/predictor.py    │      │ database/          │
     │ loads saved model  │      │ helpdesk.db        │
     └─────────┬──────────┘      └────────────────────┘
               │
               ▼
     ┌────────────────────────────────────────────┐
     │ models/smartdesk_model.joblib              │
     │ TF-IDF + Multinomial Naive Bayes pipeline  │
     │ (created by ml/train_model.py)             │
     └────────────────────────────────────────────┘
```

**Request flow for a new complaint:**

1. Streamlit collects the form data.
2. `services/validation.py` checks the inputs.
3. `services/classification.py` calls `ml/predictor.py`, which loads the trained Joblib pipeline and returns the category and confidence.
4. `services/priority.py` applies keyword rules.
5. `services/response.py` selects the category response.
6. `database/db.py` inserts the ticket with a parameterized SQL query.
7. The new ticket ID is shown to the user and the dashboard immediately reflects the new record.

---

## 8. Database Design

SQLite database file: `database/helpdesk.db` (created automatically on first start).

### `tickets` table

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | Numeric ID; displayed as `TKT-000001` |
| `name` | TEXT NOT NULL | User's full name |
| `email` | TEXT NOT NULL | Validated email address |
| `subject` | TEXT NOT NULL | Complaint subject |
| `description` | TEXT NOT NULL | Full complaint description |
| `category` | TEXT NOT NULL | AI-predicted category |
| `confidence` | REAL NOT NULL | Model confidence (0 to 1) |
| `priority` | TEXT NOT NULL | Rule-based priority |
| `suggested_response` | TEXT NOT NULL | Category-based response |
| `status` | TEXT NOT NULL | `Pending`, `In Progress` or `Resolved` (CHECK constraint) |
| `created_at` | TEXT NOT NULL | Local timestamp `YYYY-MM-DD HH:MM:SS` |

Notes:

- All inserts and updates use **parameterized SQL queries** (`?` placeholders) to prevent SQL injection.
- The table uses `CREATE TABLE IF NOT EXISTS`, so restarting the app does **not** delete tickets.
- The database is initialized in `database/db.py` from `database/schema.sql`.
- The numeric ID is converted to a readable ticket code using `format_ticket_id()`.
- **Do not delete `database/helpdesk.db`** if you want to keep existing records.

---

## 9. Project Folder Structure

```text
HelpDesk-AI/
│
├── app.py                      # Streamlit UI and page navigation
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore
│
├── .streamlit/
│   └── config.toml             # Clean light theme
│
├── data/
│   └── complaints.csv          # 100 labeled complaints / 5 categories
│
├── database/
│   ├── __init__.py
│   ├── db.py                   # SQLite connection, CRUD and statistics
│   ├── schema.sql              # tickets table definition
│   └── helpdesk.db             # Generated automatically at runtime
│
├── ml/
│   ├── __init__.py
│   ├── train_model.py          # Training + evaluation script
│   └── predictor.py            # Load model and predict
│
├── models/                     # Generated by training (not committed)
│   ├── smartdesk_model.joblib  # Complete trained pipeline
│   ├── vectorizer.joblib       # TF-IDF vectorizer
│   ├── classifier.joblib       # Naive Bayes classifier
│   └── metrics.json            # Real evaluation results
│
├── services/
│   ├── __init__.py
│   ├── validation.py           # Form validation
│   ├── classification.py       # ML service + low-confidence handling
│   ├── priority.py             # Rule-based priority detection
│   └── response.py             # Category-based suggested replies
│
└── tests/
    └── test_app_logic.py       # Pytest automated tests
```

---

## 10. Installation Instructions (Windows)

### Prerequisites

- Windows 10 or 11
- Python 3.11 or newer (Python 3.12 recommended and tested)
- Internet access once, to install the Python packages

Check Python:

```bat
python --version
```

### Step 1 – Open a terminal in the project folder

In PowerShell or Command Prompt:

```bat
cd D:\Pranav_Work_Space\Repo\HelpDesk-AI
```

### Step 2 – Create a virtual environment

```bat
python -m venv venv
```

### Step 3 – Activate the virtual environment

Command Prompt:

```bat
venv\Scripts\activate
```

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell blocks activation with an execution-policy error, run this once **for the current terminal only**, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
```

### Step 4 – Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 11. How to Train the ML Model

Run this **from the project root**, with the virtual environment activated:

```bat
python -m ml.train_model
```

What the script does:

1. Loads `data/complaints.csv`.
2. Validates the dataset.
3. Splits it into 80% training and 20% testing data (`random_state=42`, stratified).
4. Fits TF-IDF only on the training text.
5. Trains Multinomial Naive Bayes.
6. Evaluates on the test set.
7. Prints accuracy, precision, recall, F1-score and the classification report.
8. Saves the model, vectorizer, classifier and metrics into `models/`.

Expected console output (values are real results; they may differ slightly if the dataset is edited):

```text
============================================================
SmartDesk AI - Model Training Complete
============================================================
Dataset records : 100
Training records: 80
Testing records : 20
Categories      : Account Issue, General Inquiry, Hardware Issue, Network Issue, Software Issue
------------------------------------------------------------
Accuracy        : 0.9500
Precision (macro): 0.9600
Recall (macro)   : 0.9500
F1-score (macro) : 0.9492
------------------------------------------------------------
Classification Report:
                 precision    recall  f1-score   support

  Account Issue       0.80      1.00      0.89         4
General Inquiry       1.00      1.00      1.00         4
 Hardware Issue       1.00      1.00      1.00         4
  Network Issue       1.00      1.00      1.00         4
  Software Issue       1.00      0.75      0.86         4

       accuracy                           0.95        20
      macro avg       0.96      0.95      0.95        20
   weighted avg       0.96      0.95      0.95        20
```

> These are the real results of the included 100-row dataset with `random_state=42`. They are not invented in this README. After retraining, the terminal and **About Project** page show the current results. You can also open `models/metrics.json`.

**Important:** The `models/` files are generated locally and excluded by `.gitignore`. Anyone who clones the project must run the training command once before using AI features.

You can also train from the UI: the sidebar shows **Train model now** when no model exists, and the About page has a **Retrain Model Now** button.

---

## 12. How to Run the Application

With the virtual environment activated, from the project root:

```bat
streamlit run app.py
```

Streamlit prints a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser. Press `Ctrl + C` in the terminal to stop the application.

The SQLite database is created automatically in `database/helpdesk.db` on first start.

### Demonstration flow for faculty

1. Open **Dashboard** and show that counts are currently zero (or reflect existing tickets).
2. Open **Submit Complaint** and submit an example:

   - Name: `Rahul Verma`
   - Email: `rahul@example.com`
   - Subject: `Wifi keeps disconnecting`
   - Description: `My laptop wifi disconnects every few minutes and I cannot access the internet.`

3. Point out:
   - the generated ticket ID (`TKT-000001`),
   - the ML-predicted category (`Network Issue`),
   - the confidence percentage,
   - the rule-based priority (`Medium` or `High`, with the matched keyword),
   - the suggested response.
4. Submit another complaint, for example an account issue:

   - Subject: `Forgot password`
   - Description: `I forgot my password and I cannot login to my account.`

5. Open **Dashboard** and show that Total, Pending, Category, Priority and Status charts have increased using real DB data.
6. Open **Manage Tickets** and:
   - search `TKT-000001`,
   - filter by category / priority / status,
   - open a ticket,
   - change its status to `In Progress`, then `Resolved`.
7. Return to **Dashboard** and show the updated status counts.
8. Restart the app (`Ctrl + C`, then `streamlit run app.py`) and show that the tickets still exist in SQLite.
9. Open **About Project** and show the real accuracy, precision, recall, F1-score, per-category report and confusion matrix.
10. Run `python -m pytest tests -v` in the terminal to show the automated tests passing.

---

## 13. Testing Instructions

Run all tests:

```bat
python -m pytest tests -v
```

The tests cover:

- Valid complaint validation.
- Missing required fields.
- Invalid email format.
- Short description rejection.
- High / Medium / Low priority rules.
- Default priority behavior.
- Suggested response for every category.
- ML classification of clear complaints in all five categories.
- Graceful handling of unknown words and empty input.
- Ticket creation and retrieval from a temporary SQLite database.
- End-to-end classification → priority → response → database flow.
- Ticket status updates.
- Invalid status rejection.
- Search by subject and by readable ticket ID.
- Category / priority / status filters.
- Dashboard statistics from real database rows.
- Ticket ID formatting and parsing.
- Dataset size, category count and missing values.

The tests use a **temporary database**, so they do not modify `database/helpdesk.db`. If the model is missing, the test session trains it once automatically.

Expected final line:

```text
======================= XX passed in X.XXs =======================
```

---

## 14. Model Evaluation and Limitations

### Evaluation method

- 80 training records, 20 held-out test records.
- Fixed `random_state=42` for reproducibility.
- Stratified split so every category is represented in the test set.
- TF-IDF fitted only on training data (no leakage).
- Macro-averaged precision, recall and F1 are reported so each category counts equally.

### Actual evaluation results

Results from the included dataset (80 train / 20 test, `random_state=42`):

| Metric | Value |
|---|---|
| Accuracy | **95.00%** |
| Precision (macro) | **96.00%** |
| Recall (macro) | **95.00%** |
| F1-score (macro) | **94.92%** |

| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Account Issue | 0.80 | 1.00 | 0.89 | 4 |
| General Inquiry | 1.00 | 1.00 | 1.00 | 4 |
| Hardware Issue | 1.00 | 1.00 | 1.00 | 4 |
| Network Issue | 1.00 | 1.00 | 1.00 | 4 |
| Software Issue | 1.00 | 0.75 | 0.86 | 4 |

One of the four software test complaints was classified as an Account Issue, which is why software recall is 0.75. These metrics come from a single 20-record test set, so each test record is worth 5 percentage points. They demonstrate that the pipeline works, but they are **not** a guarantee of real-world accuracy.

Run `python -m ml.train_model` to reproduce them. The app reads `models/metrics.json`; no metric is hardcoded in the UI.

### Dataset limitation

The dataset contains **100 manually written demo complaints**, roughly 20 per category. It is intentionally small and uses clean, textbook-like English. Real users write with spelling mistakes, mixed languages, slang, incomplete sentences, attachments and multiple problems in one ticket.

Therefore:

- The reported accuracy is **not a claim of real-world accuracy**.
- A production system would need hundreds or thousands of representative, anonymised tickets.
- The categories and priority rules would need review by actual support staff.
- The model should be retrained periodically as new complaint patterns appear.

### Other limitations

- No user authentication: the admin pages are open on the local machine. This is acceptable for a local academic demo but not for deployment.
- Single-label classification only: one ticket receives one category.
- Priority is keyword-based and can miss paraphrased urgency; it defaults to `Medium` when no rule matches.
- Suggested responses are generic first-line replies, not full resolutions.
- SQLite is suitable for a local demo, not for many concurrent users.
- Low-confidence handling routes tickets to `General Inquiry` for manual review; an administrator should re-check them.
- Streamlit is designed for data apps; it is not a full production ticketing platform.

---

## 15. Future Enhancements

- Add administrator login with hashed passwords and role-based access.
- Replace keyword priority with a trained priority classifier once enough labeled data exists.
- Support multi-label classification for tickets describing several issues.
- Add email notifications to users when a ticket status changes.
- Add ticket comments, assignment to support staff and resolution notes.
- Add a feedback loop where administrators correct categories and the model is retrained.
- Export tickets to CSV/Excel for reporting.
- Use a larger, anonymised real help desk dataset and cross-validation.
- Improve text preprocessing for spelling mistakes and multilingual complaints.
- Deploy to an internal server with a production database if needed.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `python` is not recognized (or opens Microsoft Store) | Install Python from python.org and enable "Add Python to PATH", restart the terminal, or use the `py` launcher: `py -3.12 -m venv venv`. This project already has a working `venv`. |
| `streamlit` is not recognized | Activate the virtual environment and run `pip install -r requirements.txt`. |
| Package downloads are very slow | Try an alternate mirror: `pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/` |
| "Trained model not found" | Run `python -m ml.train_model` from the project root. |
| Import errors such as `No module named 'database'` | Run commands from the project root (`HelpDesk-AI`), not from inside `ml/` or `tests/`. |
| PowerShell will not activate `venv` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |
| Tickets disappeared | The database file may have been deleted. Do not delete `database/helpdesk.db`. |
| Charts are empty | Submit at least one complaint; charts are built only from actual database records. |
| Tests fail because the model is missing | Run `python -m ml.train_model` first, then `python -m pytest tests -v`. |

---

## Quick Command Reference

```bat
cd D:\Pranav_Work_Space\Repo\HelpDesk-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m ml.train_model
python -m pytest tests -v
streamlit run app.py
```

---

## Academic Note

This project demonstrates a complete, honest Machine Learning workflow: a labeled dataset, a proper train/test split, TF-IDF feature extraction, Multinomial Naive Bayes classification, real evaluation metrics, confidence reporting, graceful uncertainty handling, persistent storage, and automated tests. The priority rules and suggested responses are clearly separated from the ML model so their behavior can be explained and audited.

**SmartDesk AI is a demonstration and learning project, not a production help desk platform.**
