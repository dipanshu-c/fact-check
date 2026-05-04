# Fact Check

A Flask-based fact verification application that combines machine learning, natural language processing, and source credibility analysis to help users verify news headlines and claims.

## 🚀 Project Summary

`Fact Check` is designed to detect fake news and unreliable claims using a trained machine learning model, engineered text features, and source credibility checks. The application includes a web frontend, a Python backend, and MongoDB-based persistence for users, saved analyses, and credibility sources.

## ✨ Key Features

- Flask web application with a modern interface in `templates/index.html`
- JWT-based user authentication and protected API endpoints
- MongoDB integration for users, analyses, reviews, and verified sources
- Machine learning model training using TF-IDF and custom engineered features
- Content cleaning, uncertainty scoring, and source credibility estimation
- Production-ready deployment support via `Gunicorn` and `Procfile`

## 📁 Repository Structure

- `app.py` — Flask application and backend logic
- `templates/index.html` — main frontend page
- `train_model.py` — model training and saving pipeline
- `final_dataset.csv` — dataset used for training the fake news detector
- `models/` — saved model artifacts
- `requirements.txt` — Python dependencies
- `Procfile` — Gunicorn launch configuration for deployment

## 🧠 Technology Stack

- Python 3
- Flask
- MongoDB (`pymongo`)
- Flask-JWT-Extended
- Scikit-learn, NumPy, SciPy
- Requests, feedparser
- Gunicorn for production deployment

## ⚙️ Prerequisites

- Python 3.8+ installed
- `pip` package manager
- MongoDB database available and reachable via `MONGO_URI`
- Optional: virtual environment for dependency isolation

## 📦 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dipanshu-c/fact-check.git
   cd fact-check-main
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with at least:
   ```env
   MONGO_URI=<your-mongodb-connection-string>
   JWT_SECRET_KEY=<your-secret-key>
   ```

## ▶️ Running the Application

Start the Flask app locally:
```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## 📊 Training the Model

Train the fake news classifier and save the model artifact:
```bash
python train_model.py
```

This script will:
- Load `final_dataset.csv`
- Build TF-IDF and engineered features
- Train a calibrated logistic regression classifier
- Save the trained model to `models/fakenews_model.pkl`

## 📝 Environment Variables

- `MONGO_URI` — MongoDB connection string
- `JWT_SECRET_KEY` — secret key used for JWT tokens

## 🚀 Deployment

For production deployment, the app can be started with Gunicorn using `Procfile`:
```bash
gunicorn app:app
```

## 🛠️ Notes

- The app initializes default verified sources in MongoDB if none exist
- `EngineeredFeatures` extracts signals for sensational language, expert references, quotes, and source indicators
- Model training uses a balanced logistic regression with probability calibration for better confidence scoring
- If present, `models/fakenews_model.pkl` is loaded on startup and used by the hybrid prediction system
- A local Hugging Face transformer model can optionally be used by setting `BERT_FAKE_NEWS_MODEL`

## 🧩 API Endpoints

### Authentication
- `POST /api/auth/register`
  - Request: `{ "email": "user@example.com", "password": "secret" }`
  - Response: registration success and `userId`
- `POST /api/auth/login`
  - Request: `{ "email": "user@example.com", "password": "secret" }`
  - Response: JWT token and user details
- `GET /api/auth/verify`
  - Requires Authorization header with Bearer token
  - Response: token validity and basic user info

### Analysis & Prediction
- `POST /api/analyze`
  - Requires JWT auth
  - Request: `{ "headline": "...", "content": "...", "url": "...", "category": "..." }`
  - If only `url` is provided, the app attempts to scrape title and content from the page
  - Response: verdict, confidence, model/heuristic scores, factor breakdown, and community consensus
- `POST /api/analyze/explain`
  - Requires JWT auth
  - Request: `{ "headline": "...", "content": "..." }`
  - Response: detailed feature factors and top contributors to the prediction

### Reviews & Community Feedback
- `POST /api/reviews`
  - Requires JWT auth
  - Request: `{ "analysisId": "...", "verdict": "real|fake|uncertain", "text": "..." }`
  - Response: review creation status
- `GET /api/reviews/<analysis_id>`
  - Response: list of reviews for a given analysis
- `GET /api/reviews/<analysis_id>/consensus`
  - Response: community consensus summary for an analysis

### User Data
- `GET /api/user/analyses`
  - Requires JWT auth
  - Response: all analyses created by the current user
- `GET /api/user/reviews`
  - Requires JWT auth
  - Response: all reviews created by the current user
- `GET /api/user/analytics`
  - Requires JWT auth
  - Response: basic user analytics and review accuracy estimate

### Sources & Admin
- `GET /api/sources`
  - Response: list of verified source domains in MongoDB
- `POST /api/sources`
  - Requires JWT auth and admin user
  - Request: `{ "name": "Example News", "url": "example.com", "category": "news" }`
  - Response: new source creation success
- `GET /api/admin/analytics`
  - Requires JWT auth and admin user
  - Response: system-wide analytics counts
- `GET /api/admin/users`
  - Requires JWT auth and admin user
  - Response: list of registered users and their activity counts

## 🧠 Model & Prediction Architecture

### Model Training
- Training script: `train_model.py`
- Data source: `final_dataset.csv`
- Features:
  - TF-IDF over combined headline and content text
  - Engineered features from `EngineeredFeatures` for sensationalism, credibility, emotional tone, controversy, and coherence
- Classifier:
  - `LogisticRegression` with `class_weight='balanced'`
  - wrapped in `CalibratedClassifierCV` for calibrated probability scores
- Saved artifact: `models/fakenews_model.pkl`

### Runtime Prediction
- `app.py` loads `models/fakenews_model.pkl` when available
- Prediction pipeline uses `PickleLoadedModel` to:
  - transform inputs with TF-IDF and engineered features
  - apply the calibrated classifier
  - fall back to score-based heuristics if the saved model is unavailable
- The hybrid system blends:
  - ML score from the trained model
  - heuristic score from `HeuristicFakeNewsModel`
  - optional local transformer score from `BERT_FAKE_NEWS_MODEL`
  - source credibility boost based on trusted domains

### Heuristic Fallback
- `HeuristicFakeNewsModel` extracts interpretable signals such as:
  - source credibility and trustworthiness
  - sensational language and emotional wording
  - expert references and quoted evidence
  - controversy and multi-source indications
- It returns a synthetic verdict and confidence even without a trained model

### Community Feedback
- The `/api/analyze` endpoint can blend machine predictions with community review consensus when enough reviews exist
- This adds a crowd-sourced signal into the final verdict and confidence score

## 🤝 Contributing

Contributions are welcome! Feel free to:
- open issues for bugs or improvement ideas
- submit pull requests for feature updates
- enhance model training, source validation, or UI experience


## 📌 Author

- Maintainer: `dipanshu-c`

---

**Last Updated**: May 4, 2026
