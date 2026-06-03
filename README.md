# SMS Spam Detection

A Python machine-learning project for classifying SMS (and similar short text) messages as spam or not spam. The repository includes a trained TF‑IDF + classifier pipeline (serialized with pickle) and a Streamlit web app that supports both single-message prediction and batch prediction from uploaded files.

## Features

- Spam vs. Not Spam classification for short text messages
- Streamlit web interface
  - Single message prediction from a text area
  - Batch prediction from uploaded `.txt` or `.csv` files
  - Downloadable CSV output containing predictions
- Text preprocessing using NLTK tokenization, stopword removal, and stemming
- Pre-trained artifacts included:
  - `vectorizer.pkl` (TF‑IDF vectorizer)
  - `model.pkl` (trained classifier)
- Training notebook included (`SMS_spam_detection.ipynb`) for experimentation and retraining

## Project Structure

```
.
├── README.md                 # Project documentation
├── SMS_spam_detection.ipynb  # Notebook for training/evaluation
├── app.py                    # Streamlit application (UI + inference)
├── requirements.txt          # Python dependencies
├── setup.py                  # Packaging metadata (optional)
├── model.pkl                 # Serialized trained model
├── vectorizer.pkl            # Serialized TF-IDF vectorizer
└── src/
    └── __init__.py
```

## Requirements and Dependencies

- Python 3.9+ recommended
- Packages (see `requirements.txt`):
  - numpy
  - pandas
  - streamlit
  - seaborn
  - matplotlib
  - scikit-learn

Additional runtime requirement:
- NLTK datasets: `punkt`, `stopwords`  
  The Streamlit app attempts to download these at runtime.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/preettrank53/SMS_spam_detaction.git
   cd SMS_spam_detaction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   ```

   Linux/macOS:

   ```bash
   source venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. (Optional) Pre-download NLTK resources to avoid runtime downloads:

   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

### 1. Run the Streamlit App (Recommended)

Start the web app:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

#### Single Message Prediction
- Paste a message into the input area
- Click **Predict** to classify it as Spam / Not Spam

#### Batch Prediction
Upload one of the following:
- `.txt` file: one message per line
- `.csv` file: must contain a `message` column

After processing, the app will:
- show per-row predictions
- provide a “final decision” for the combined text
- allow downloading a CSV with the predictions

### 2. Notebook (Training / Experimentation)

Open the notebook:

```bash
jupyter notebook SMS_spam_detection.ipynb
```

Use it to explore the dataset, train models, tune parameters, and (if implemented in the notebook) export updated `model.pkl` and `vectorizer.pkl`.

### 3. Programmatic Inference (Using the Pickle Artifacts)

If you want to run predictions in your own Python script, you can load the serialized artifacts and apply the same preprocessing approach used in `app.py`.

Example:

```python
import pickle
import nltk
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Ensure NLTK resources are available
nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text: str) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words("english") and t not in string.punctuation]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

message = "Congratulations! You have won a prize. Click the link to claim."
x = tfidf.transform([transform_text(message)])
pred = model.predict(x)[0]

print("Spam" if pred == 1 else "Not Spam")
```

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-change
   ```
3. Make your changes (keep code readable and add docstrings/comments where helpful)
4. Run the app locally to verify behavior:
   ```bash
   streamlit run app.py
   ```
5. Commit and push:
   ```bash
   git commit -m "Describe your change"
   git push origin feature/your-change
   ```
6. Open a Pull Request

## Author / Contact

Maintained by **Preet Rank**.  
Email: **preetrank53@gmail.com**
