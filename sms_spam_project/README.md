# 📱 SMS Spam Classifier

A **robust, production-grade Python CLI** for classifying SMS messages as spam or ham using a pre-trained scikit-learn model.  
Supports **streaming CSV processing**, **parallel CPU inference**, **progress bars**, and **throughput benchmarking**.

---

## 🚀 Features

✅ Always includes **spam probability** (confidence score)  
✅ **Schema validation** — fails fast if the text column is missing  
✅ **Parallel processing** with per-worker model loading  
✅ **Streaming CSV processing** (handles millions of rows)  
✅ **Atomic writes** to prevent corruption on failure  
✅ **Progress bar** with rows/sec tracking (Windows/Linux compatible)  
✅ **Graceful interrupt handling** (`Ctrl+C`)  
✅ **Benchmark mode** for throughput measurement  
✅ **Structured logging** to console  

---

## 📂 Project Structure

```
sms_spam_project/
├── sms_classify.py        # CLI entry point (main script)
├── sms_classifier/        # Core package
│   ├── __init__.py
│   ├── batching.py        # Batch splitting helper
│   ├── io_utils.py        # Chunked CSV reader
│   ├── model_loader.py    # Model loading + inference wrapper
│   └── predictor.py       # Core classification logic
├── model.pkl              # Pre-trained model (TfidfVectorizer + LogisticRegression)
├── input_messages.csv     # Example input file
├── predictions.csv        # Example output file
└── README.md              # Documentation
```

---

## 🧩 Installation

```bash
pip install -r requirements.txt
```

### Requirements

- `scikit-learn==1.1.3` → must match the model’s training version  
- `pandas`
- `tqdm`
- `colorama` (for Windows progress bar rendering)
- `joblib`

If you encounter version mismatch warnings from scikit-learn, install the same version used during training:

```bash
pip install scikit-learn==1.1.3
```

---

## ⚙️ Usage

Run from the command line:

```bash
python sms_classify.py   --input input_messages.csv   --output predictions.csv   --model model.pkl   --workers 4   --batch-size 4096   --chunksize 100000   --benchmark
```

### 💡 Example (Windows)

```bash
python sms_classify.py --input big_input.csv --output big_predictions.csv --model model.pkl --workers 4 --benchmark
```

---

## 🧾 Command-Line Arguments

| Argument | Required | Default | Description |
|-----------|-----------|----------|--------------|
| `--input`, `-i` | ✅ | — | Path to input CSV (must contain text column). |
| `--output`, `-o` | ✅ | — | Path to output predictions CSV. |
| `--model`, `-m` |  | `model.pkl` | Path to pre-trained model file. |
| `--text-col` |  | `message` | Name of the text column. |
| `--chunksize` |  | `5000` | Rows per streaming chunk. |
| `--batch-size` |  | `512` | Number of messages per prediction batch. |
| `--workers` |  | `0` | Number of worker processes (0 = single process). |
| `--benchmark` |  | `False` | Print throughput (rows/sec) at the end. |
| `--force` |  | `False` | Overwrite existing output file. |
| `--log-level` |  | `INFO` | Logging verbosity (`DEBUG`, `INFO`, etc.). |

🟢 **Note:** Probability (`spam_probability`) is **always output** — the old `--proba` flag is now deprecated.

---

## 📊 Example Output

```csv
id,message,prediction,spam_probability
1,"Congratulations! You have been selected for a free cruise.",1,0.92
2,"Hey, are we still meeting at 6?",0,0.03
3,"Limited offer! Claim your $1000 gift card now.",1,0.97
```

- `prediction`: `1` = spam, `0` = ham  
- `spam_probability`: Model confidence score between `0.0` and `1.0`

---

## ⏱️ Benchmarking Example

### Command

```bash
python sms_classify.py   --input big_input.csv   --output predictions.csv   --model model.pkl   --workers 4   --batch-size 4096   --chunksize 100000   --benchmark
```

### Example Output

```
Classifying chunks: 100%|███████████████████████████████████████| 10/10 [00:05<00:00, 2.46chunk/s, rows=50000, rps=12272.5]
2025-10-04 21:48:45,457 INFO Wrote 50000 predictions to 'predictions.csv'.
2025-10-04 21:48:45,458 INFO Done.
```

### Example Performance (local CPU)

| Workers | Batch Size | Rows  | Time (s) | Rows/sec |
|---------|------------|-------|----------|----------|
| 2       |      4096  |50,000 |     8.6  | 9318     |
| 4       |      4096  |50,000 |     8.6  | 10410    |
| 6       |      4096  |50,000 |     8.6  | 10790    |
| 8       |      4096  |50,000 |     8.6  | 10417    |


---

## 🧠 Design Highlights

- **Streaming pipeline** → handles large datasets efficiently.  
- **Atomic writes** → ensures output integrity (temporary file replaced atomically).  
- **Cross-platform progress visualization** → powered by `tqdm.auto` + `colorama`.  
- **Fail-fast schema validation** → detects missing or incorrect text column names and suggests fixes.  
- **Parallelism safety** → each worker loads its own model (avoids pickle sharing issues).  
- **Clean structured logging** → INFO and DEBUG modes supported.  

---

## 🧭 Development Notes

---

## 🧩 Future Enhancements

- ✅ Config dataclass for pipeline configuration  
- 🔜 FastAPI / Streamlit inference service  
- 🔜 Structured JSON logging  
- 🔜 Cloud-ready (S3 / GCS I/O)  
- 🔜 Checkpointing & resumable processing  
- 🔜 CI/CD pipeline for packaging & deployment  

---

## 🧰 Troubleshooting

### ⚠️ InconsistentVersionWarning

If you see:
```
Trying to unpickle estimator from version 1.1.3 when using version 1.7.2
```
Your local scikit-learn version differs from the one used during training.

**Fix:**
```bash
pip install scikit-learn==1.1.3
```

---

## 👨‍💻 Developer Setup (Optional)

To work on this project locally:

```bash
# Clone repository
git clone https://github.com/yourname/sms_spam_project.git
cd sms_spam_project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run tests (if applicable)

```bash
pytest -v
```

---

## 🧾 License

This project is licensed under the **MIT License** — feel free to modify and distribute.

---

## 🏁 Summary

This CLI tool offers a **fast, memory-safe, and production-ready SMS spam classification pipeline**  
that can process massive datasets in parallel, with live progress visualization and robust error handling.

> 🧠 Ideal for integration into larger ETL, data quality, or message moderation pipelines.
