# app.py
import os
import re
import traceback
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv

# machine-learning related imports used by loaded artifact
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JWT_SECRET_KEY'] = os.getenv(
    'JWT_SECRET_KEY',
    'factcheck-dipanshu-2005-11-07-secret-jwt-key-8171484821-2025-11-26'
)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

jwt = JWTManager(app)

# --- MongoDB Connection (fixed) ---
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/fakenews_detector')
client = MongoClient(MONGO_URI)

# get_default_database() raises if you try to use it in a boolean test.
# Call it once and compare against None. If the URI includes a database name,
# get_default_database() returns that Database; otherwise fall back to a named DB.
try:
    default_db = client.get_default_database()
except Exception:
    default_db = None

db = default_db if default_db is not None else client['fakenews_detector']

users_collection = db['users']
analyses_collection = db['analyses']
reviews_collection = db['reviews']
sources_collection = db['sources']
# --- end DB init ---


# Initialize verified sources if empty
def init_sources():
    if sources_collection.count_documents({}) == 0:
        default_sources = [
            {'name': 'BBC News', 'url': 'bbc.com', 'category': 'news', 'verified': True, 'createdAt': datetime.now()},
            {'name': 'Reuters', 'url': 'reuters.com', 'category': 'news', 'verified': True, 'createdAt': datetime.now()},
            {'name': 'Associated Press', 'url': 'apnews.com', 'category': 'news', 'verified': True, 'createdAt': datetime.now()},
            {'name': 'Nature', 'url': 'nature.com', 'category': 'science', 'verified': True, 'createdAt': datetime.now()},
            {'name': 'Science Daily', 'url': 'sciencedaily.com', 'category': 'science', 'verified': True, 'createdAt': datetime.now()},
            {'name': 'The Guardian', 'url': 'theguardian.com', 'category': 'news', 'verified': True, 'createdAt': datetime.now()},
        ]
        sources_collection.insert_many(default_sources)

init_sources()

# ------------------ EngineeredFeatures (needed for unpickling) ------------------
class EngineeredFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.controversial = ['conspiracy', 'hoax', 'cover-up', 'fake']
        self.emotional = ['shocking', 'unbelievable', 'outrageous', 'disgusting', 'amazing']
        self.expert = ['expert', 'scientist', 'doctor', 'professor', 'analyst']
        self.source_kw = ['study', 'research', 'university', 'confirmed', 'verified']
        self.source_phrases = ['according to', 'reports say', 'sources claim', 'authorities say']
        self.passive_indicators = ['was', 'were', 'been', 'being']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for headline, content in X:
            if headline is None: headline = ''
            if content is None: content = ''
            text = (headline + " " + content).lower()
            words = text.split()
            nwords = max(len(words), 1)

            sensationalism = (text.count('!') + text.count('?')) / nwords
            source_cred = sum(1 for kw in self.source_kw if kw in text) / max(len(self.source_kw), 1)
            numbers = len(re.findall(r'\d+', text))
            dates = len(re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text))
            specificity = (numbers + dates) / nwords
            emotional = sum(1 for w in self.emotional if w in text) / max(len(self.emotional), 1)
            passive = sum(1 for w in self.passive_indicators if w in words) / nwords
            quote_factor = min((text.count('"') + text.count("'")) / 4, 1.0)
            expert_ref = sum(1 for kw in self.expert if kw in text) / max(len(self.expert), 1)
            controversy = sum(1 for kw in self.controversial if kw in text) / max(len(self.controversial), 1)
            multi_source = sum(1 for kw in self.source_phrases if kw in text) / max(len(self.source_phrases), 1)
            headline_words = set(headline.lower().split())
            content_words = set(content.lower().split())
            coherence = len(headline_words.intersection(content_words)) / max(len(headline_words), 1)

            rows.append([
                round(min(sensationalism, 1.0), 6),
                round(min(source_cred, 1.0), 6),
                round(min(specificity, 1.0), 6),
                round(min(emotional, 1.0), 6),
                round(min(passive, 1.0), 6),
                quote_factor,
                round(min(expert_ref, 1.0), 6),
                round(min(controversy, 1.0), 6),
                round(min(multi_source, 1.0), 6),
                round(min(coherence, 1.0), 6)
            ])
        return np.array(rows, dtype=float)
# ---------------------------------------------------------------------------------

# ------------------ Community consensus helper ------------------
def calculate_community_consensus(analysis_id):
    """Calculate consensus from user reviews for an analysis."""
    try:
        analysis_obj_id = ObjectId(analysis_id)
    except Exception:
        return None

    reviews = list(reviews_collection.find({'analysisId': analysis_obj_id}))
    if not reviews:
        return None

    verdict_counts = {'real': 0, 'fake': 0, 'uncertain': 0}
    for review in reviews:
        verdict = review.get('verdict', 'uncertain')
        if verdict in verdict_counts:
            verdict_counts[verdict] += 1

    total = sum(verdict_counts.values())
    if total == 0:
        return None

    consensus = {
        'real': round((verdict_counts['real'] / total) * 100, 1),
        'fake': round((verdict_counts['fake'] / total) * 100, 1),
        'uncertain': round((verdict_counts['uncertain'] / total) * 100, 1),
        'totalReviews': total,
        'dominantVerdict': max(verdict_counts, key=verdict_counts.get)
    }
    return consensus
# ---------------------------------------------------------------------------------

# ------------------ Heuristic fallback model ------------------
class HeuristicFakeNewsModel:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def extract_features(self, headline, content):
        headline = headline or ''
        content = content or ''
        text = (headline + ' ' + content).lower()
        sensationalism = (text.count('!') + text.count('?')) / max(len(text.split()), 1)
        credible_keywords = ['study', 'research', 'university', 'confirmed', 'verified']
        source_credibility = sum(1 for kw in credible_keywords if kw in text) / len(credible_keywords)
        numbers = len(re.findall(r'\d+', text))
        dates = len(re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text))
        specificity = (numbers + dates) / max(len(text.split()), 1)
        emotional_words = ['shocking', 'unbelievable', 'outrageous', 'disgusting', 'amazing']
        emotional = sum(1 for word in emotional_words if word in text) / len(emotional_words)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive = sum(1 for word in passive_indicators if word in text.split()) / max(len(text.split()), 1)
        quotes = text.count('"') + text.count("'")
        quote_factor = min(quotes / 4, 1.0)
        expert_keywords = ['expert', 'scientist', 'doctor', 'professor', 'analyst']
        expert_reference = sum(1 for kw in expert_keywords if kw in text) / len(expert_keywords)
        controversial = ['conspiracy', 'hoax', 'cover-up', 'fake']
        controversy = sum(1 for kw in controversial if kw in text) / len(controversial)
        source_keywords = ['according to', 'reports say', 'sources claim', 'authorities say']
        multi_source = sum(1 for kw in source_keywords if kw in text) / len(source_keywords)
        headline_words = set(headline.lower().split())
        content_words = set(content.lower().split())
        coherence = len(headline_words.intersection(content_words)) / max(len(headline_words), 1)

        return {
            'sensationalism': min(sensationalism, 1.0),
            'source_credibility': min(source_credibility, 1.0),
            'specificity': min(specificity, 1.0),
            'emotional_language': min(emotional, 1.0),
            'passive_voice': min(passive, 1.0),
            'quote_presence': quote_factor,
            'expert_reference': min(expert_reference, 1.0),
            'controversy_level': min(controversy, 1.0),
            'multi_source': min(multi_source, 1.0),
            'coherence': coherence
        }

    def predict(self, headline, content):
        f = self.extract_features(headline, content)
        lstm_score = np.mean([f['source_credibility'], f['specificity'], f['coherence'], 1 - f['sensationalism'], f['multi_source']]) * 100
        cgpnn_score = np.mean([f['expert_reference'], f['quote_presence'], f['multi_source'], 1 - f['controversy_level'], f['coherence']]) * 100
        ml_score = (lstm_score * 0.6 + cgpnn_score * 0.4)
        if ml_score >= 70:
            verdict = 'real'
            confidence = ml_score
        elif ml_score <= 40:
            verdict = 'fake'
            confidence = 100 - ml_score
        else:
            verdict = 'uncertain'
            confidence = 50
        return {
            'verdict': verdict,
            'confidence': int(confidence),
            'mlScore': round(ml_score, 1),
            'lstmScore': round(lstm_score, 1),
            'cgpnnScore': round(cgpnn_score, 1),
            'factors': {k: round(v, 3) for k, v in f.items()}
        }
# ---------------------------------------------------------------------------------

# ------------------ Robust Pickle loader to handle multiple artifact shapes ------------------
MODEL_PATH = os.path.join('models', 'fakenews_model.pkl')

class PickleLoadedModel:
    def __init__(self, artifact):
        self.tfidf = None
        self.eng = None
        self.scaler = None
        self.clf = None
        self.vectorizer = None
        self.model = None
        self.label_classes = None
        self.label_encoder = None

        if isinstance(artifact, dict):
            # try to pick fields sensibly
            self.tfidf = artifact.get('tfidf') or artifact.get('vectorizer')
            self.eng = artifact.get('eng')
            self.scaler = artifact.get('scaler')
            self.clf = artifact.get('clf') or artifact.get('model')
            self.vectorizer = artifact.get('vectorizer') if self.tfidf is None else None
            self.model = artifact.get('model') if self.clf is None else None
            self.label_classes = artifact.get('label_classes')
            self.label_encoder = artifact.get('label_encoder')
        else:
            self.clf = artifact

    def _get_positive_index(self, classes=None):
        if classes:
            try:
                lower_classes = [str(c).strip().lower() for c in classes]
                if 'real' in lower_classes:
                    return lower_classes.index('real')
                if '1' in lower_classes and '0' in lower_classes:
                    return lower_classes.index('1')
                return len(lower_classes) - 1
            except Exception:
                pass
        return 1

    def predict(self, headline, content):
        combined = (headline or '') + " " + (content or '')

        # TFIDF + engineered features + scaler + clf
        if self.tfidf is not None and self.eng is not None and self.scaler is not None and self.clf is not None:
            tf = self.tfidf.transform([combined])
            eng = self.eng.transform([(headline or '', content or '')])
            X = hstack([tf, self.scaler.transform(eng)])

            classes = None
            if self.label_classes:
                classes = self.label_classes
            elif self.label_encoder is not None:
                try:
                    classes = list(self.label_encoder.classes_)
                except Exception:
                    classes = None

            pos_index = self._get_positive_index(classes)

            try:
                proba_arr = self.clf.predict_proba(X)[0]
                if pos_index < 0 or pos_index >= len(proba_arr):
                    pos_index = int(np.argmax(proba_arr))
                proba = float(proba_arr[pos_index]) * 100.0
            except Exception:
                try:
                    score = self.clf.decision_function(X)[0]
                    proba = 1 / (1 + np.exp(-score)) * 100
                except Exception:
                    proba = 50.0

            verdict = 'real' if proba >= 60 else ('fake' if proba <= 40 else 'uncertain')
            confidence = int(round(proba if verdict == 'real' else (100 - proba) if verdict == 'fake' else 50))
            return {
                'verdict': verdict,
                'confidence': confidence,
                'mlScore': round(proba, 1),
                'lstmScore': None,
                'cgpnnScore': None,
                'factors': {}
            }

        # vectorizer + model fallback
        if self.vectorizer is not None and self.model is not None:
            X = self.vectorizer.transform([combined])
            try:
                proba_arr = self.model.predict_proba(X)[0]
                classes = None
                if self.label_classes:
                    classes = self.label_classes
                elif self.label_encoder is not None:
                    try:
                        classes = list(self.label_encoder.classes_)
                    except Exception:
                        classes = None
                pos_index = self._get_positive_index(classes)
                if pos_index < 0 or pos_index >= len(proba_arr):
                    pos_index = int(np.argmax(proba_arr))
                proba = float(proba_arr[pos_index]) * 100.0
            except Exception:
                try:
                    score = self.model.decision_function(X)[0]
                    proba = 1 / (1 + np.exp(-score)) * 100
                except Exception:
                    proba = 50.0

            verdict = 'real' if proba >= 60 else ('fake' if proba <= 40 else 'uncertain')
            confidence = int(round(proba if verdict == 'real' else (100 - proba) if verdict == 'fake' else 50))
            return {
                'verdict': verdict,
                'confidence': confidence,
                'mlScore': round(proba, 1),
                'lstmScore': None,
                'cgpnnScore': None,
                'factors': {}
            }

        # last-resort classifier that accepts raw text
        if self.clf is not None:
            try:
                proba_arr = self.clf.predict_proba([combined])[0]
                pos_index = 1
                if hasattr(self.clf, 'classes_') and len(self.clf.classes_) > 1:
                    try:
                        pos_index = list(self.clf.classes_).index(1) if 1 in list(self.clf.classes_) else 1
                    except Exception:
                        pos_index = 1
                if pos_index < 0 or pos_index >= len(proba_arr):
                    pos_index = int(np.argmax(proba_arr))
                proba = float(proba_arr[pos_index]) * 100.0
            except Exception:
                try:
                    score = self.clf.decision_function([combined])[0]
                    proba = 1 / (1 + np.exp(-score)) * 100
                except Exception:
                    proba = 50.0

            verdict = 'real' if proba >= 60 else ('fake' if proba <= 40 else 'uncertain')
            confidence = int(round(proba if verdict == 'real' else (100 - proba) if verdict == 'fake' else 50))
            return {
                'verdict': verdict,
                'confidence': confidence,
                'mlScore': round(proba, 1),
                'lstmScore': None,
                'cgpnnScore': None,
                'factors': {}
            }

        return {'verdict': 'uncertain', 'confidence': 50, 'mlScore': 50.0, 'lstmScore': None, 'cgpnnScore': None, 'factors': {}}

# Attempt to load artifact
ml_model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            artifact = pickle.load(f)
        ml_model = PickleLoadedModel(artifact)
        print("[INFO] Loaded trained model from", MODEL_PATH)
    except Exception as e:
        print("[WARN] Failed to load model artifact:", e)
        traceback.print_exc()
        ml_model = None

# fallback to heuristic
if ml_model is None:
    print("[INFO] Using fallback heuristic model (no trained artifact found or failed to load).")
    ml_model = HeuristicFakeNewsModel()
# ---------------------------------------------------------------------------------

# ------------------ Prediction with community feedback ------------------
def predict_with_community_feedback(headline, content, analysis_id=None):
    ml_prediction = ml_model.predict(headline, content)

    community_consensus = None
    if analysis_id:
        community_consensus = calculate_community_consensus(analysis_id)

    if not community_consensus or community_consensus.get('totalReviews', 0) < 2:
        return ml_prediction, community_consensus

    ml_score = ml_prediction.get('mlScore', 50)
    community_score = (
        community_consensus['real'] +
        (community_consensus['uncertain'] * 0.5) +
        (community_consensus['fake'] * 0)
    )

    blended_score = (ml_score * 0.6) + (community_score * 0.4)

    if blended_score >= 70:
        final_verdict = 'real'
        final_confidence = blended_score
    elif blended_score <= 40:
        final_verdict = 'fake'
        final_confidence = 100 - blended_score
    else:
        final_verdict = 'uncertain'
        final_confidence = 50

    enhanced_prediction = ml_prediction.copy()
    enhanced_prediction['verdict'] = final_verdict
    enhanced_prediction['confidence'] = int(final_confidence)
    enhanced_prediction['mlScore'] = round(blended_score, 1)
    enhanced_prediction['communityInfluence'] = {
        'blendedScore': round(blended_score, 1),
        'consensus': community_consensus,
        'weights': {'ml': 0.6, 'community': 0.4}
    }

    return enhanced_prediction, community_consensus
# ---------------------------------------------------------------------------------

# ==================== Authentication Routes ====================
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    if users_collection.find_one({'email': email}):
        return jsonify({'error': 'User already exists'}), 400

    user = {
        'email': email,
        'password': generate_password_hash(password),
        'isAdmin': False,
        'createdAt': datetime.now(),
        'reviewCount': 0,
        'analysisCount': 0
    }

    result = users_collection.insert_one(user)
    return jsonify({'success': True, 'message': 'Registration successful', 'userId': str(result.inserted_id)}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    user = users_collection.find_one({'email': email})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    access_token = create_access_token(
        identity=str(user['_id']),
        additional_claims={'email': email, 'isAdmin': user.get('isAdmin', False)}
    )
    return jsonify({'token': access_token, 'userId': str(user['_id']), 'email': email, 'isAdmin': user.get('isAdmin', False)}), 200

@app.route('/api/auth/verify', methods=['GET'])
@jwt_required()
def verify_token():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    user = users_collection.find_one({'_id': user_obj_id})
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'valid': True,
        'user': {'email': user['email'], 'userId': str(user['_id']), 'joinDate': user['createdAt'].strftime('%m/%d/%Y')},
        'isAdmin': user.get('isAdmin', False)
    }), 200

# ==================== Analysis Routes ====================
@app.route('/api/analyze', methods=['POST'])
@jwt_required()
def analyze_news():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    data = request.get_json() or {}
    headline = data.get('headline', '') or ''
    content = data.get('content', '') or ''
    url = data.get('url', '') or ''
    category = data.get('category', 'general') or 'general'

    if not headline and not content and not url:
        return jsonify({'error': 'No content provided'}), 400

    # run ML model
    prediction = ml_model.predict(headline, content)

    analysis = {
        'userId': user_obj_id,
        'headline': headline,
        'content': content,
        'url': url,
        'category': category,
        'verdict': prediction.get('verdict'),
        'confidence': int(prediction.get('confidence') or 0),
        'mlScore': float(prediction.get('mlScore') or 0.0),
        'lstmScore': prediction.get('lstmScore', None),
        'cgpnnScore': prediction.get('cgpnnScore', None),
        'factors': prediction.get('factors') or {},
        'communityVote': {'real': 0, 'fake': 0, 'uncertain': 0},
        'communityInfluence': None,
        'createdAt': datetime.utcnow()
    }

    result = analyses_collection.insert_one(analysis)
    users_collection.update_one({'_id': user_obj_id}, {'$inc': {'analysisCount': 1}})

    return jsonify({
        'analysisId': str(result.inserted_id),
        'headline': headline,
        'content': content,
        'url': url,
        'verdict': prediction.get('verdict'),
        'confidence': int(prediction.get('confidence') or 0),
        'mlScore': float(prediction.get('mlScore') or 0.0),
        'lstmScore': prediction.get('lstmScore', None),
        'cgpnnScore': prediction.get('cgpnnScore', None),
        'factors': prediction.get('factors') or {},
        'communityVote': analysis['communityVote'],
        'createdAt': analysis['createdAt'].isoformat()
    }), 201

from operator import itemgetter


@app.route('/api/analyze/explain', methods=['POST'])
@jwt_required()
def explain_analysis():
    data = request.get_json() or {}
    headline = data.get('headline', '')
    content = data.get('content', '')


    # run model (or heuristic) to get factors
    pred = ml_model.predict(headline, content)
    factors = pred.get('factors') or {}


    # If factors empty and we have EngineeredFeatures available, compute them
    try:
        eng = EngineeredFeatures()
        feat_arr = eng.transform([(headline, content)])[0]
        feat_names = ['sensationalism','source_cred','specificity','emotional','passive','quote_factor','expert_ref','controversy','multi_source','coherence']
        factors = dict(zip(feat_names, [float(round(float(x), 3)) for x in feat_arr]))
    except Exception:
        pass


    # top 3 contributors by magnitude (absolute)
    top3 = sorted(factors.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    return jsonify({'factors': factors, 'topContributors': top3, 'rawPrediction': pred}), 200


# Create a new review (logged-in users only) — robust version
@app.route('/api/reviews', methods=['POST'])
@jwt_required()
def create_review():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    data = request.get_json() or {}
    analysis_id = data.get('analysisId') or None
    verdict = (data.get('verdict') or 'uncertain').lower()
    text = (data.get('text') or '').strip()

    if verdict not in ('real', 'fake', 'uncertain'):
        verdict = 'uncertain'

    analysis_obj_id = None
    if analysis_id:
        try:
            analysis_obj_id = ObjectId(analysis_id)
            # optional: check existence
            if not analyses_collection.find_one({'_id': analysis_obj_id}):
                # don't fail — log and continue, we still keep the review
                print(f"[WARN] create_review: analysis id provided but not found: {analysis_id}")
                analysis_obj_id = None
        except Exception:
            print(f"[WARN] create_review: invalid analysis id format: {analysis_id}")
            analysis_obj_id = None

    review_doc = {
        'userId': user_obj_id,
        'analysisId': analysis_obj_id,    # may be None
        'verdict': verdict,
        'text': text,
        'helpful': 0,
        'createdAt': datetime.utcnow()
    }

    try:
        result = reviews_collection.insert_one(review_doc)
        users_collection.update_one({'_id': user_obj_id}, {'$inc': {'reviewCount': 1}})
        # update aggregate counters on analysis if valid
        if analysis_obj_id:
            try:
                analyses_collection.update_one(
                    {'_id': analysis_obj_id},
                    {'$inc': {f'communityVote.{verdict}': 1}}
                )
            except Exception as e:
                print(f"[WARN] Failed to update analysis.communityVote: {e}")

        return jsonify({
            'success': True,
            'reviewId': str(result.inserted_id),
            # return canonical timestamp property name 'timestamp' to match server responses
            'timestamp': review_doc['createdAt'].isoformat()
        }), 201
    except Exception as e:
        print("[ERROR] Failed to insert review:", e)
        traceback.print_exc()
        return jsonify({'error': 'Failed to create review'}), 500


@app.route('/api/reviews/<analysis_id>', methods=['GET'])
def get_reviews(analysis_id):
    try:
        analysis_obj_id = ObjectId(analysis_id)
    except Exception:
        return jsonify({'error': 'Invalid analysis id'}), 400

    revs = list(reviews_collection.find({'analysisId': analysis_obj_id}).sort('createdAt', 1))
    out = []
    for r in revs:
        user_str = 'Anonymous'
        try:
            if r.get('userId'):
                u = users_collection.find_one({'_id': ObjectId(r.get('userId'))})
                if u:
                    user_str = u.get('email', 'Anonymous')
        except Exception:
            pass
        out.append({
            'id': str(r.get('_id')),
            'user': user_str,
            'verdict': r.get('verdict', 'uncertain'),
            'text': r.get('text', ''),
            'helpful': r.get('helpful', 0),
            'timestamp': r.get('createdAt').isoformat() if r.get('createdAt') else None
        })

    return jsonify({'reviews': out}), 200

@app.route('/api/reviews/<analysis_id>/consensus', methods=['GET'])
def get_consensus(analysis_id):
    try:
        analysis_obj_id = ObjectId(analysis_id)
    except Exception:
        return jsonify({'error': 'Invalid analysis id'}), 400

    consensus = calculate_community_consensus(analysis_id)
    if not consensus:
        return jsonify({'consensus': None, 'message': 'No reviews yet'}), 200
    return jsonify({'consensus': consensus}), 200

# ==================== User analyses endpoint (persistent history) ====================
@app.route('/api/user/analyses', methods=['GET'])
@jwt_required()
def get_user_analyses():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    cursor = list(analyses_collection.find({'userId': user_obj_id}).sort('createdAt', -1))
    results = []
    for a in cursor:
        aid = a.get('_id')
        # get sample reviews (latest 5)
        rev_docs = list(reviews_collection.find({'analysisId': aid}).sort('createdAt', -1).limit(5))
        sample_reviews = []
        for r in rev_docs:
            username = 'Anonymous'
            try:
                if r.get('userId'):
                    u = users_collection.find_one({'_id': ObjectId(r.get('userId'))})
                    if u:
                        username = u.get('email', 'Anonymous')
            except Exception:
                pass
            sample_reviews.append({
                'id': str(r.get('_id')),
                'user': username,
                'verdict': r.get('verdict'),
                'text': r.get('text'),
                'timestamp': r.get('createdAt').isoformat() if r.get('createdAt') else None
            })

        consensus = calculate_community_consensus(str(aid))
        results.append({
            'analysisId': str(a.get('_id')),
            'headline': a.get('headline'),
            'content': a.get('content'),
            'url': a.get('url'),
            'verdict': a.get('verdict'),
            'confidence': a.get('confidence'),
            'mlScore': a.get('mlScore'),
            'communityVote': a.get('communityVote', {}),
            'communityConsensus': consensus,
            'sampleReviews': sample_reviews,
            'createdAt': a.get('createdAt').isoformat() if a.get('createdAt') else None
        })
    return jsonify({'analyses': results}), 200


@app.route('/api/user/reviews', methods=['GET'])
@jwt_required()
def user_reviews_endpoint():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    docs = list(reviews_collection.find({'userId': user_obj_id}).sort('createdAt', -1))
    out = []
    for r in docs:
        analysis_summary = None
        try:
            aid = r.get('analysisId')
            if aid:
                a = analyses_collection.find_one({'_id': aid})
                if a:
                    analysis_summary = {
                        'analysisId': str(a['_id']),
                        'headline': a.get('headline'),
                        'verdict': a.get('verdict'),
                        'confidence': a.get('confidence')
                    }
        except Exception:
            analysis_summary = None

        out.append({
            'id': str(r.get('_id')),
            'analysisId': str(r.get('analysisId')) if r.get('analysisId') else None,
            'verdict': r.get('verdict'),
            'text': r.get('text'),
            'timestamp': r.get('createdAt').isoformat() if r.get('createdAt') else None,
            'analysis': analysis_summary
        })
    return jsonify({'reviews': out}), 200

# ==================== Sources & analytics (unchanged, kept for compatibility) ====================
@app.route('/api/sources', methods=['GET'])
def get_sources():
    sources = list(sources_collection.find())
    return jsonify({
        'sources': [{
            'id': str(s['_id']),
            'name': s['name'],
            'url': s['url'],
            'category': s.get('category', ''),
            'verified': s.get('verified', False)
        } for s in sources]
    }), 200

@app.route('/api/sources', methods=['POST'])
@jwt_required()
def add_source():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    user = users_collection.find_one({'_id': user_obj_id})
    if not user or not user.get('isAdmin'):
        return jsonify({'error': 'Admin access required'}), 403

    data = request.get_json() or {}
    name = data.get('name')
    url = data.get('url')
    category = data.get('category', '')

    if not name or not url:
        return jsonify({'error': 'Name and URL required'}), 400

    source = {'name': name, 'url': url, 'category': category, 'verified': True, 'createdAt': datetime.now()}
    result = sources_collection.insert_one(source)
    return jsonify({'success': True, 'sourceId': str(result.inserted_id)}), 201

@app.route('/api/user/analytics', methods=['GET'])
@jwt_required()
def user_analytics():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    user = users_collection.find_one({'_id': user_obj_id})
    user_analyses = list(analyses_collection.find({'userId': user_obj_id}))
    user_reviews = list(reviews_collection.find({'userId': user_obj_id}))

    correct = sum(1 for r in user_reviews if r['verdict'] in ['real', 'fake'])
    accuracy = (correct / len(user_reviews) * 100) if user_reviews else 0

    return jsonify({
        'totalAnalyses': len(user_analyses),
        'totalReviews': len(user_reviews),
        'accuracyScore': round(accuracy, 1),
        'joinDate': user['createdAt'].isoformat() if user and 'createdAt' in user else None
    }), 200

@app.route('/api/admin/analytics', methods=['GET'])
@jwt_required()
def admin_analytics():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    user = users_collection.find_one({'_id': user_obj_id})
    if not user or not user.get('isAdmin'):
        return jsonify({'error': 'Admin access required'}), 403

    total_users = users_collection.count_documents({})
    total_analyses = analyses_collection.count_documents({})
    total_reviews = reviews_collection.count_documents({})

    real_analyses = analyses_collection.count_documents({'verdict': 'real'})
    fake_analyses = analyses_collection.count_documents({'verdict': 'fake'})
    total = real_analyses + fake_analyses
    accuracy = (max(real_analyses, fake_analyses) / total * 100) if total > 0 else 0

    return jsonify({
        'totalUsers': total_users,
        'totalAnalyses': total_analyses,
        'totalReviews': total_reviews,
        'systemAccuracy': round(accuracy, 1)
    }), 200

@app.route('/api/admin/users', methods=['GET'])
@jwt_required()
def get_all_users():
    current_user_id = get_jwt_identity()
    try:
        user_obj_id = ObjectId(current_user_id)
    except Exception:
        return jsonify({'error': 'Invalid user id'}), 400

    user = users_collection.find_one({'_id': user_obj_id})
    if not user or not user.get('isAdmin'):
        return jsonify({'error': 'Admin access required'}), 403

    users = list(users_collection.find())
    return jsonify({
        'users': [{
            'email': u['email'],
            'joinDate': u.get('createdAt').isoformat() if u.get('createdAt') else None,
            'reviewCount': u.get('reviewCount', 0),
            'analysisCount': u.get('analysisCount', 0),
            'userId': str(u.get('_id'))
        } for u in users]
    }), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
