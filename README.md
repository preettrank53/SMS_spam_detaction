SMS Spam Detection SystemA high-performance machine learning pipeline and user interface designed to analyze, classify, and filter short message service (SMS) communications. This project implements advanced Natural Language Processing (NLP) techniques and supervised machine learning classifiers to determine whether incoming text messages are legitimate (ham) or unsolicited (spam).Project OverviewUnsolicited text messages pose a persistent challenge in mobile telecommunications, often carrying security, privacy, and financial risks. This repository provides an end-to-end machine learning solution to address this problem. The system processes raw text data, extracts highly informative contextual features, and executes binary classification to predict the probability of a message being spam.Mathematical ApproachFeature extraction is primarily accomplished using the Term Frequency-Inverse Document Frequency (TF-IDF) representation, which penalizes highly frequent words across the corpus to emphasize specific contextual keywords.The mathematical formulation for Term Frequency (TF) of a term $t$ within a document $d$ is defined as:$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$Where $f_{t,d}$ represents the raw frequency of the term within the document. The Inverse Document Frequency (IDF) is defined as:$$\text{IDF}(t, D) = \log \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$Where $|D|$ is the total number of documents in the corpus. The combined weight is calculated as:$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$For classification, the system supports multiple algorithms, with Multinomial Naive Bayes as the baseline. It operates on the assumption of conditional independence among features, applying Bayes' Theorem:$$P(C = c \mid \mathbf{x}) \propto P(C = c) \prod_{i=1}^{n} P(x_i \mid C = c)$$Where $C$ represents the class label (spam or ham) and $\mathbf{x} = [x_1, x_2, \dots, x_n]$ represents the tokenized feature vector of the input SMS.FeaturesComplete natural language preprocessing pipeline, including lowercasing, tokenization, alphanumeric filtering, stop-word removal, and stemming or lemmatization.Feature extraction using TF-IDF vectorization with adjustable n-gram ranges and sublinear term-frequency scaling.Training routines for multiple classifier architectures, including Multinomial Naive Bayes, Support Vector Machines (SVM), and Logistic Regression.Evaluation metrics generation featuring confusion matrices, Precision, Recall, and F1-Score calculations.Modular prediction pipeline allowing single-string inference as well as batch inference.Minimalist web interface built using Streamlit, facilitating real-time text validation.Project StructureSMS_spam_detaction/
├── data/
│   └── spam.csv                     # The raw dataset containing labeled messages
├── models/
│   ├── classifier.pkl               # Serialized trained machine learning model
│   └── vectorizer.pkl               # Serialized TF-IDF vectorizer configuration
├── notebooks/
│   └── spam_detection_training.ipynb# Interactive model training and evaluation notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # Text normalization and feature extraction module
│   └── predict.py                   # Model inference utilities
├── app.py                           # Streamlit-based web application dashboard
├── requirements.txt                 # Manifest of environment dependencies
└── README.md                        # Documentation
Requirements and DependenciesThe system is engineered using Python 3.9 or higher. The specific package requirements are detailed below:numpy (Core numerical computation)pandas (Structured dataset handling)scikit-learn (Text vectorization, training algorithms, and evaluations)nltk (Natural language tokenization, corpus filtering, and stemming)joblib or pickle (Model preservation and deserialization)streamlit (User interface dashboard)Installation InstructionsFollow these steps to configure a local development environment.1. Clone the RepositoryClone the project folder from GitHub using git:git clone [https://github.com/preettrank53/SMS_spam_detaction.git](https://github.com/preettrank53/SMS_spam_detaction.git)
cd SMS_spam_detaction
2. Configure a Virtual EnvironmentIt is recommended to use a clean virtual environment to prevent package version conflicts:# Create the environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
3. Install DependenciesInstall the necessary python modules through pip:pip install --upgrade pip
pip install -r requirements.txt
4. Download Language ResourcesCertain NLTK helper libraries (such as tokenizers or stop-word tables) must be initialized locally. Execute the following command within a python terminal:import nltk
nltk.download('punkt')
nltk.download('stopwords')
Usage1. Model TrainingTo retrain the machine learning model on your target dataset:Open the interactive notebook located in the notebooks directory:jupyter notebook notebooks/spam_detection_training.ipynb
Alternatively, run the preprocessing and training script directly:python src/preprocess.py
This procedure generates or updates the serialized model files (classifier.pkl and vectorizer.pkl) inside the models/ directory.2. Programmatic InferenceTo programmatically integrate the classifier into an existing Python application, load the serialized pipelines using the following implementation pattern:import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Configure preprocessing utilities
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Deserialization of saved ML pipeline components
try:
    vectorizer = joblib.load('models/vectorizer.pkl')
    model = joblib.load('models/classifier.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Trained model or vectorizer binaries are missing. Please execute training first.")

# Sample prediction logic
def predict_sms(message):
    processed_message = clean_text(message)
    vectorized_input = vectorizer.transform([processed_message])
    prediction = model.predict(vectorized_input)[0]
    probability = model.predict_proba(vectorized_input)[0]
    
    label = "Spam" if prediction == 1 else "Ham"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return {
        "classification": label,
        "confidence_score": round(float(confidence), 4)
    }

# Execution
example_text = "Congratulations! You have won a 1,000 USD cash prize. Claim your reward immediately by clicking this link."
result = predict_sms(example_text)
print(f"Message Status: {result['classification']} (Confidence: {result['confidence_score'] * 100}%)")
3. Launching the Web InterfaceRun the Streamlit application to inspect predictions through an interactive user interface:streamlit run app.py
The application will launch on your local host (typically http://localhost:8501). Enter any text into the provided text-area element to inspect classification and probability outputs instantaneously.ContributingWe welcome contributions to optimize models, add new feature extractors, or enhance the dashboard interface.Fork the repository.Create a specific feature branch for your changes:git checkout -b feature/optimized-classifier
Commit your implementations. Ensure clean coding standards and relevant docstring usage:git commit -m "Refactor TF-IDF parameters and add SVM option"
Push your changes to the remote branch:git push origin feature/optimized-classifier
Initiate a formal Pull Request targeting the primary branch.LicenseThis software project is licensed under the MIT License. You may inspect the complete legal terms in the LICENSE file situated in the repository root.Contact and AttributionFor queries, technical discussions, or to report vulnerabilities, please contact:Author: Rank PreetGitHub: preettrank53Repository URL: https://github.com/preettrank53/SMS_spam_detaction
