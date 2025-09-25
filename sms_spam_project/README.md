# SMS Spam Classifier

A robust Python CLI application for classifying SMS messages as spam or ham using a pre-trained scikit-learn model.

## 📂 Project Structure
```
sms_spam_project/
├── sms_classify.py        # CLI entry point
├── sms_classifier/        # Package with helpers
│   ├── __init__.py
│   ├── batching.py
│   ├── io_utils.py
│   ├── model_loader.py
│   └── predictor.py
├── model.pkl              # Pre-trained model (TfidfVectorizer + LogisticRegression)
├── input_messages.csv     # Example input file
├── predictions.csv        # Example output file
└── README.md              # Documentation
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements include:
- scikit-learn==1.1.3 (match model training version)
- joblib
- pandas

## Usage

Run from the command line:

```bash
python sms_classify.py   --input input_messages.csv   --output predictions.csv   --model model.pkl   --workers 4   --batch-size 4096   --chunksize 100000   --proba   --benchmark
```

### Arguments
- `--input`: Input CSV file with messages.
- `--output`: Output CSV file with predictions.
- `--model`: Path to `model.pkl`.
- `--workers`: Number of parallel worker processes (default=0, i.e., single process).
- `--batch-size`: Number of messages per prediction batch.
- `--chunksize`: Number of rows per input chunk when streaming CSV.
- `--proba`: If set, output spam probabilities.
- `--benchmark`: Measure and print throughput (rows/sec).

### Example Output (with `--proba`)
```
id,message,prediction,spam_probability
1,"Congratulations! You have been selected for a free cruise.",1,0.92
2,"Hey, are we still meeting at 6?",0,0.03
```

---

## Benchmarking

Performance will vary depending on hardware and environment.  
The following results were obtained on a **local development laptop** (CPU-based inference, no GPU).

### Example Command
```bash
python sms_classify.py   --input big_input.csv   --output big_predictions.csv   --model model.pkl   --workers 4   --batch-size 4096   --chunksize 100000   --proba   --benchmark
```

### Results (50,000 rows, local machine)
| Workers | Batch Size | Chunksize | Proba | Rows | Time (s) | Rows/sec |
|---------|------------|-----------|-------|------|----------|----------|
| 4       | 4096       | 100,000   | ❌ No | 50,000 | 4.34 | **11,516** |
| 4       | 4096       | 100,000   | ✅ Yes | 50,000 | 8.68 | **5,761** |

### Observations
- **Without `--proba`**: Faster, only outputs binary labels (`0=ham, 1=spam`).  
- **With `--proba`**: Slower, but adds `spam_probability` (confidence score). Useful for threshold tuning, monitoring, and explainability.  
- **Small inputs** (e.g., <100 rows) appear slower because startup overhead dominates.  
- Throughput scales with batch size and workers until CPU saturation.  

### Notes
- Use `--workers` ≈ number of CPU cores.  
- Larger `--batch-size` (2k–8k) improves throughput.  
- Larger `--chunksize` (50k–200k) reduces I/O overhead.  
- Ensure `scikit-learn==1.1.3` to avoid version mismatch warnings with the provided model.  

---

## Next Steps
- Preload model into shared memory for parallel workers.  
- Expose real-time service mode (e.g., FastAPI).  
- Add monitoring/logging (Prometheus, structured logs).  
- CI/CD pipeline for packaging and deployment.  
- Explore GPU acceleration for future scaling.  
- Event-triggered deployment (e.g., AWS Lambda/GCP Function): automatically run classification when a new file arrives in storage.  
