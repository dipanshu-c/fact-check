# app.py
import os
import re
import html
import traceback
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import certifi
import requests
import time
from threading import Lock
import feedparser
import numpy as np
from urllib.parse import urlparse
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
load_dotenv()
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin

load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['JWT_SECRET_KEY'] = os.getenv(
    'JWT_SECRET_KEY',
    'factcheck-dipanshu-2005-11-07-secret-jwt-key-8171484821-2025-11-26'
)
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=30)

jwt = JWTManager(app)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    logger.error("MONGO_URI environment variable not set. Exiting.")
    raise Exception("MONGO_URI environment variable not set")



mongo_options = {
    "serverSelectionTimeoutMS": 10000, 
    "connectTimeoutMS": 10000,
    "socketTimeoutMS": None,
}

try:
    mongo_options["tls"] = True
    mongo_options["tlsCAFile"] = certifi.where()

    client = MongoClient(MONGO_URI, **mongo_options)
    client.admin.command("ping")
    logger.info("Connected to MongoDB successfully.")
except Exception as e:
    logger.exception("Failed to connect to MongoDB with provided MONGO_URI.")
    raise

try:
    default_db = client.get_default_database()
except Exception:
    default_db = None

db = default_db if default_db is not None else client['fakenews_detector']

users_collection = db['users']
analyses_collection = db['analyses']
reviews_collection = db['reviews']
sources_collection = db['sources']

try:
    users_collection.create_index("email", unique=True)
except Exception:
    logger.exception("Failed to create one or more DB indexes.")

def init_sources():
    if sources_collection.count_documents({}) == 0:
        default_sources = [
            {'name': 'BBC News', 'url': 'bbc.com', 'category': 'news', 'verified': True, 'createdAt': datetime.utcnow()},
            {'name': 'Reuters', 'url': 'reuters.com', 'category': 'news', 'verified': True, 'createdAt': datetime.utcnow()},
            {'name': 'Associated Press', 'url': 'apnews.com', 'category': 'news', 'verified': True, 'createdAt': datetime.utcnow()},
            {'name': 'Nature', 'url': 'nature.com', 'category': 'science', 'verified': True, 'createdAt': datetime.utcnow()},
            {'name': 'Science Daily', 'url': 'sciencedaily.com', 'category': 'science', 'verified': True, 'createdAt': datetime.utcnow()},
            {'name': 'The Guardian', 'url': 'theguardian.com', 'category': 'news', 'verified': True, 'createdAt': datetime.utcnow()},
        ]
        sources_collection.insert_many(default_sources)
init_sources()

def check_source_credibility(url):
    if not url:
        return 0.5

    for source in sources_collection.find({"verified": True}):
        if source["url"] in url:
            return 1.0

    return 0.3

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

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def calculate_uncertainty(proba):
        p = proba / 100
        entropy = - (p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9))
        return entropy

class TextPreprocessor:
    def normalize(self, headline, content):
        headline = html.unescape(headline or '')
        content = html.unescape(content or '')
        text = f"{headline} {content}"
        text = re.sub(r'https?://\S+', ' URL ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return headline.strip(), content.strip(), text

    def tokens(self, text):
        return re.findall(r"\b[a-z][a-z'-]*\b", text.lower())


class OptionalTransformerScorer:
    """Uses a locally available BERT-style fake-news model when configured.

    Set BERT_FAKE_NEWS_MODEL to a local Hugging Face model directory. The app
    intentionally uses local_files_only=True so startup never downloads a model.
    """
    def __init__(self):
        self.ready = False
        self.tokenizer = None
        self.model = None
        self.labels = []
        model_path = os.getenv("BERT_FAKE_NEWS_MODEL", "").strip()
        if not model_path:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            self.model.eval()
            id2label = getattr(self.model.config, "id2label", {}) or {}
            self.labels = [str(id2label.get(i, i)).lower() for i in range(self.model.config.num_labels)]
            self.ready = True
            logger.info("Loaded local transformer model from %s", model_path)
        except Exception as e:
            logger.warning("BERT_FAKE_NEWS_MODEL is set but could not be loaded locally: %s", e)

    def real_score(self, text):
        if not self.ready:
            return None
        try:
            import torch
            inputs = self.tokenizer(
                text[:4000],
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits[0]
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            real_idx = None
            fake_idx = None
            for i, label in enumerate(self.labels):
                if any(word in label for word in ("real", "true", "reliable", "legit")):
                    real_idx = i
                if any(word in label for word in ("fake", "false", "misleading", "rumor")):
                    fake_idx = i

            if real_idx is not None:
                return float(probs[real_idx] * 100)
            if fake_idx is not None:
                return float((1 - probs[fake_idx]) * 100)
        except Exception as e:
            logger.warning("Transformer scoring failed: %s", e)
        return None


TRANSFORMER_SCORER = OptionalTransformerScorer()

TRUSTED_SOURCE_DOMAINS = {
    'reuters.com', 'apnews.com', 'associatedpress.com', 'bbc.com', 'bbc.co.uk',
    'nytimes.com', 'theguardian.com', 'aljazeera.com', 'npr.org', 'pbs.org',
    'thehindu.com', 'indianexpress.com', 'hindustantimes.com', 'business-standard.com',
    'nature.com', 'science.org', 'who.int', 'un.org', 'gov.in', 'nic.in'
}

def source_context_score(source_url):
    if not source_url:
        return 0.0, False
    try:
        host = urlparse(source_url).netloc.lower()
        host = host[4:] if host.startswith('www.') else host
        trusted = any(host == domain or host.endswith('.' + domain) for domain in TRUSTED_SOURCE_DOMAINS)
        return (1.0 if trusted else 0.25), trusted
    except Exception:
        return 0.0, False

class HeuristicFakeNewsModel:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            stop_words='english'
        )

    def extract_features(self, headline, content, source_url=''):
        headline, content, raw_text = TextPreprocessor().normalize(headline, content)
        text = raw_text.lower()
        words = TextPreprocessor().tokens(text)
        nwords = max(len(words), 1)
        source_score, trusted_source = source_context_score(source_url)
        sentence_count = len(re.findall(r'[.!?]+(?:\s|$)', raw_text))

        sensationalism = (raw_text.count('!') + raw_text.count('?')) / nwords
        credible_keywords = [
            'study', 'research', 'university', 'confirmed', 'verified',
            'official', 'statement', 'court', 'ministry', 'agency', 'reuters',
            'associated press', 'bbc', 'according to'
        ]
        source_credibility = sum(1 for kw in credible_keywords if kw in text) / len(credible_keywords)
        numbers = len(re.findall(r'\d+', text))
        dates = len(re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b', text))
        specificity = (numbers + dates) / nwords
        emotional_words = [
            'shocking', 'unbelievable', 'outrageous', 'disgusting', 'amazing',
            'miracle', 'secret', 'exposed', 'banned', 'viral'
        ]
        emotional = sum(1 for word in emotional_words if word in text) / len(emotional_words)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive = sum(1 for word in passive_indicators if word in words) / nwords
        quotes = raw_text.count('"') + raw_text.count("'")
        quote_factor = min(quotes / 4, 1.0)
        expert_keywords = ['expert', 'scientist', 'doctor', 'professor', 'analyst', 'official', 'spokesperson']
        expert_reference = sum(1 for kw in expert_keywords if kw in text) / len(expert_keywords)
        controversial = [
            'conspiracy', 'hoax', 'cover-up', 'fake', 'rumor', 'rumour',
            'doctors hate', 'hidden truth', 'share before', 'wake up'
        ]
        controversy = sum(1 for kw in controversial if kw in text) / len(controversial)
        source_keywords = ['according to', 'reported by', 'authorities say', 'officials said', 'court said']
        multi_source = sum(1 for kw in source_keywords if kw in text) / len(source_keywords)
        headline_words = set(TextPreprocessor().tokens(headline))
        content_words = set(TextPreprocessor().tokens(content))
        coherence = len(headline_words.intersection(content_words)) / max(len(headline_words), 1)
        evidence = min(source_credibility + specificity + quote_factor + expert_reference + multi_source + source_score * 0.35, 1.0)
        risk = min(sensationalism + emotional + controversy, 1.0)
        article_context = min(
            (1 if nwords >= 35 else nwords / 35) * 0.5
            + min(sentence_count / 4, 1.0) * 0.2
            + min((numbers + dates) / 3, 1.0) * 0.15
            + min((quote_factor + expert_reference + multi_source + source_credibility), 1.0) * 0.15,
            1.0
        )

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
            'coherence': min(coherence, 1.0),
            'evidence_strength': evidence,
            'fake_risk': risk,
            'article_context': article_context,
            'source_score': source_score,
            'trusted_source': 1.0 if trusted_source else 0.0,
            'word_count': float(nwords)
        }

    def predict(self, headline, content, source_url=''):
        f = self.extract_features(headline, content, source_url)
        evidence_score = np.mean([
            f['source_credibility'],
            f['specificity'],
            f['quote_presence'],
            f['expert_reference'],
            f['multi_source'],
            f['coherence']
        ]) * 100
        risk_score = np.mean([
            f['sensationalism'],
            f['emotional_language'],
            f['controversy_level'],
            1 - f['coherence']
        ]) * 100
        ml_score = float(np.clip(50 + evidence_score * 0.7 - risk_score * 0.9, 1, 99))

        has_evidence = f['evidence_strength'] >= 0.14 or f['article_context'] >= 0.72 or f['trusted_source'] >= 1.0
        high_risk = f['fake_risk'] >= 0.16
        if high_risk and ml_score < 55:
            verdict = 'fake'
            confidence = max(65, 100 - ml_score)
        elif has_evidence and ml_score >= 62:
            verdict = 'real'
            confidence = ml_score
        elif ml_score <= 42:
            verdict = 'fake'
            confidence = 100 - ml_score
        else:
            verdict = 'uncertain'
            confidence = int(abs(ml_score - 50) * 2)
            confidence = int(np.clip(confidence, 0, 100))
        return {
            'verdict': verdict,
            'confidence': int(confidence),
            'mlScore': round(ml_score, 1),
            'lstmScore': round(evidence_score, 1),
            'cgpnnScore': round(100 - risk_score, 1),
            'factors': {k: round(v, 3) for k, v in f.items()}
        }

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

    def _get_label_classes(self):
        if self.label_classes is not None:
            return [str(c).strip().lower() for c in self.label_classes]
        if self.label_encoder is not None:
            try:
                return [str(c).strip().lower() for c in self.label_encoder.classes_]
            except Exception:
                return None
        return None

    def _real_index_from_estimator(self, estimator):
        label_classes = self._get_label_classes()
        estimator_classes = list(getattr(estimator, 'classes_', []))

        for i, encoded_class in enumerate(estimator_classes):
            try:
                encoded_index = int(encoded_class)
                if label_classes and 0 <= encoded_index < len(label_classes):
                    label = label_classes[encoded_index]
                    if label in ('real', 'true', 'truth', 'genuine'):
                        return i
                elif str(encoded_class).strip().lower() in ('real', 'true', 'truth', 'genuine', '1'):
                    return i
            except Exception:
                label = str(encoded_class).strip().lower()
                if label in ('real', 'true', 'truth', 'genuine', '1'):
                    return i

        if label_classes and 'real' in label_classes:
            return label_classes.index('real')
        return 1 if len(estimator_classes) > 1 else 0

    def _heuristic_real_score(self, headline, content, source_url=''):
        analysis = self._heuristic_analysis(headline, content, source_url)
        return analysis['score']

    def _heuristic_analysis(self, headline, content, source_url=''):
        f = HeuristicFakeNewsModel().extract_features(headline, content, source_url)
        headline, content, raw_text = TextPreprocessor().normalize(headline, content)
        text = raw_text.lower()
        credible_phrases = [
            'according to', 'confirmed', 'official', 'reported by', 'published',
            'research', 'study', 'university', 'data', 'statement', 'reuters',
            'associated press', 'bbc', 'authorities', 'commission', 'nasa',
            'scientists', 'scientist', 'peer-reviewed', 'measurements', 'announces',
            'confirms', 'said', 'reported', 'released', 'issued', 'court',
            'ministry', 'government', 'police', 'hospital', 'agency',
            'spokesperson', 'document', 'records', 'evidence', 'survey'
        ]
        sensational_phrases = [
            'shocking', 'unbelievable', 'miracle', 'secret', 'doctors hate',
            'government cover-up', 'cover-up', 'conspiracy', 'hoax', 'viral post',
            'you will not believe', 'exposed', 'banned', 'hidden truth',
            'wake up', 'share before', 'mainstream media', '100% true',
            'guaranteed', 'must share', 'they do not want you to know',
            'breaking shocking', 'cure overnight', 'no evidence'
        ]

        credible_hits = sum(1 for phrase in credible_phrases if phrase in text)
        sensational_hits = sum(1 for phrase in sensational_phrases if phrase in text)
        words = re.findall(r'\b[a-z]{2,}\b', text)
        evidence_strength = f.get('evidence_strength', 0)
        fake_risk = f.get('fake_risk', 0)
        has_content = len(words) >= 12
        article_like = f.get('article_context', 0) >= 0.72 and fake_risk < 0.12
        trusted_source = f.get('trusted_source', 0) >= 1.0
        has_evidence_gate = (
            evidence_strength >= 0.14
            or article_like
            or trusted_source
            or credible_hits >= 2
            or (credible_hits >= 1 and (f['specificity'] > 0.02 or f['quote_presence'] > 0))
        )
        fake_gate = (
            fake_risk >= 0.16
            or sensational_hits >= 2
            or (sensational_hits >= 1 and not has_evidence_gate)
            or f['sensationalism'] >= 0.12
        )

        score = 50
        score += f['source_credibility'] * 45
        score += f['specificity'] * 140
        score += f['expert_reference'] * 35
        score += f['multi_source'] * 45
        score += f['coherence'] * 20
        score += f.get('article_context', 0) * 18
        score += f.get('source_score', 0) * 20
        score += min(credible_hits, 5) * 8
        if has_content and has_evidence_gate and fake_risk < 0.12:
            score += 6
        score -= f['sensationalism'] * 70
        score -= f['emotional_language'] * 45
        score -= f['controversy_level'] * 65
        score -= min(sensational_hits, 5) * 10

        if trusted_source and not fake_gate:
            score = max(score, 68)

        if fake_gate:
            score = min(score, 45)
        elif not has_evidence_gate:
            score = min(score, 60)

        return {
            'score': float(np.clip(score, 1, 99)),
            'factors': f,
            'hasEvidence': has_evidence_gate,
            'fakeRisk': fake_gate,
            'articleLike': article_like,
            'trustedSource': trusted_source,
            'credibleHits': credible_hits,
            'sensationalHits': sensational_hits,
            'transformerScore': TRANSFORMER_SCORER.real_score(raw_text)
        }

    def _prediction_from_real_score(self, real_score, factors=None):

        confidence = 0

        if real_score >= 62:
            verdict = 'real'
            confidence = real_score

        elif real_score <= 38:
            verdict = 'fake'
            confidence = 100 - real_score

        elif 45 <= real_score <= 55:
            verdict = 'uncertain'
            confidence = int(abs(real_score - 50) * 2)

        else:
            if real_score > 50:
                verdict = 'lean_real'
            else:
                verdict = 'lean_fake'

            confidence = int(abs(real_score - 50) * 2)

        confidence = int(np.clip(confidence, 0, 100))

        return {
            'verdict': verdict,
            'confidence': confidence,
            'mlScore': round(float(real_score), 1),
            'lstmScore': None,
            'cgpnnScore': None,
            'factors': factors or {}
        }
    
    def _calibrated_prediction(self, headline, content, model_real_score, factors=None, source_url=''):
        heuristic = self._heuristic_analysis(headline, content, source_url)
        heuristic_real_score = heuristic['score']
        transformer_score = heuristic.get('transformerScore')
        disagreement = abs(model_real_score - heuristic_real_score)

        if transformer_score is not None:
            real_score = (model_real_score * 0.2) + (heuristic_real_score * 0.35) + (transformer_score * 0.45)
        elif disagreement >= 30:
            real_score = (model_real_score * 0.15) + (heuristic_real_score * 0.85)
        else:
            real_score = (model_real_score * 0.6) + (heuristic_real_score * 0.4)

        if heuristic['fakeRisk'] and model_real_score < 55:
            real_score = min(real_score, 40)
        elif heuristic.get('trustedSource'):
            real_score = max(real_score, 68)
        elif heuristic.get('articleLike') and real_score >= 54:
            real_score = max(real_score, 60)
        elif not heuristic['hasEvidence']:
            real_score = min(real_score, 56)

        merged_factors = factors or heuristic.get('factors') or {}
        result = self._prediction_from_real_score(real_score, factors=merged_factors)
        result['modelScore'] = round(float(model_real_score), 1)
        result['heuristicScore'] = round(float(heuristic_real_score), 1)
        result['bertScore'] = round(float(transformer_score), 1) if transformer_score is not None else None
        result['hasEvidence'] = bool(heuristic['hasEvidence'])
        result['fakeRisk'] = bool(heuristic['fakeRisk'])
        result['articleLike'] = bool(heuristic.get('articleLike'))
        result['trustedSource'] = bool(heuristic.get('trustedSource'))
        return result

    def predict(self, headline, content, source_url=''):
        combined = (headline or '') + " " + (content or '')
        if self.tfidf is not None and self.eng is not None and self.scaler is not None and self.clf is not None:
            tf = self.tfidf.transform([combined])
            eng = self.eng.transform([(headline or '', content or '')])
            X = hstack([tf, self.scaler.transform(eng)])

            try:
                proba_arr = self.clf.predict_proba(X)[0]
                real_index = self._real_index_from_estimator(self.clf)
                if real_index < 0 or real_index >= len(proba_arr):
                    real_index = int(np.argmax(proba_arr))
                proba = float(proba_arr[real_index]) * 100.0
            except Exception:
                try:
                    score = self.clf.decision_function(X)[0]
                    proba = 1 / (1 + np.exp(-score)) * 100
                except Exception:
                    proba = 50.0

            return self._calibrated_prediction(headline, content, proba, source_url=source_url)

        if self.vectorizer is not None and self.model is not None:
            X = self.vectorizer.transform([combined])
            try:
                proba_arr = self.model.predict_proba(X)[0]
                real_index = self._real_index_from_estimator(self.model)
                if real_index < 0 or real_index >= len(proba_arr):
                    real_index = int(np.argmax(proba_arr))
                proba = float(proba_arr[real_index]) * 100.0
            except Exception:
                try:
                    score = self.model.decision_function(X)[0]
                    proba = 1 / (1 + np.exp(-score)) * 100
                except Exception:
                    proba = 50.0

            return self._calibrated_prediction(headline, content, proba, source_url=source_url)

        if self.clf is not None:
            try:
                proba_arr = self.clf.predict_proba([combined])[0]
                real_index = self._real_index_from_estimator(self.clf)
                if real_index < 0 or real_index >= len(proba_arr):
                    real_index = int(np.argmax(proba_arr))
                proba = float(proba_arr[real_index]) * 100.0
            except Exception:
                try:
                    score = self.clf.decision_function([combined])[0]
                    proba = 1 / (1 + np.exp(-score)) * 100
                except Exception:
                    proba = 50.0

            return self._calibrated_prediction(headline, content, proba, source_url=source_url)

        return {'verdict': 'uncertain', 'confidence': 50, 'mlScore': 50.0, 'lstmScore': None, 'cgpnnScore': None, 'factors': {}}

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

class HybridFakeNewsSystem:
    def __init__(self, ml_model, heuristic_model):
        self.ml_model = ml_model
        self.heuristic_model = heuristic_model

    def predict(self, headline, content, source_url=""):
        if self.ml_model:
            ml_pred = self.ml_model.predict(headline, content, source_url=source_url)
        else:
            ml_pred = {'mlScore': 50, 'confidence': 50}
        heur_pred = self.heuristic_model.predict(headline, content, source_url=source_url)

        ml_score = ml_pred.get("mlScore", 50)
        heur_score = heur_pred.get("mlScore", 50)

        ml_weight = 0.7 if ml_pred.get("confidence", 50) > 60 else 0.5
        heur_weight = 1 - ml_weight

        final_score = (ml_score * ml_weight) + (heur_score * heur_weight)

        final_score = 50 + (final_score - 50) * 1.3
        final_score = np.clip(final_score, 0, 100)

        if abs(ml_score - heur_score) < 10:
            if final_score > 50:
                final_score += 5
            else:
                final_score -= 5

        source_boost, _ = source_context_score(source_url)
        final_score = (final_score * 0.85) + (source_boost * 100 * 0.25)

        if final_score >= 60:
            verdict = "real"
        elif final_score <= 40:
            verdict = "fake"
        else:
            verdict = "uncertain"

        confidence = int(abs(final_score - 50) * 2)
        confidence = int(np.clip(confidence, 0, 100))

        return {
            "verdict": verdict,
            "confidence": confidence,
            "mlScore": round(final_score, 2),
            "modelScore": ml_score,
            "heuristicScore": heur_score,
            "weights": {
                "ml": ml_weight,
                "heuristic": heur_weight
            },
            "agreement": abs(ml_score - heur_score),
            "reason": {
                "ml_vs_heuristic_gap": abs(ml_score - heur_score),
                "dominant_signal": "ml" if ml_score > heur_score else "heuristic"
            }
        }
    
heuristic_model = HeuristicFakeNewsModel()
hybrid_model = HybridFakeNewsSystem(ml_model, heuristic_model)

def predict_with_community_feedback(headline, content, analysis_id=None, source_url=''):
    ml_prediction = hybrid_model.predict(headline, content, source_url=source_url)

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
        uncertainty = calculate_uncertainty(blended_score)
        final_confidence = int(np.clip((1 - uncertainty) * 100, 0, 100))

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

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')
_news_cache = {}
_news_cache_lock = Lock()
_NEWS_CACHE_TTL = int(os.getenv("NEWS_CACHE_TTL", 30))  # seconds

def clean_html(raw_html):
    if not raw_html:
        return ''
    clean = re.sub('<.*?>', '', raw_html)
    return clean.strip()

def extract_url_content(url):
    """Fetch a URL and extract enough page text for the local model."""
    if not url or not re.match(r'^https?://', url.strip(), re.I):
        return '', ''

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }
    response = requests.get(url.strip(), headers=headers, timeout=10)
    response.raise_for_status()

    html = response.text or ''
    title = ''
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, flags=re.I | re.S)
    if title_match:
        title = clean_html(title_match.group(1))

    meta_desc = ''
    meta_match = re.search(
        r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\'](.*?)["\']',
        html,
        flags=re.I | re.S
    )
    if meta_match:
        meta_desc = clean_html(meta_match.group(1))

    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I | re.S)
    paragraph_text = ' '.join(clean_html(p) for p in paragraphs[:8])
    content = (meta_desc + ' ' + paragraph_text).strip()
    return title[:300], content[:5000]
    
@app.route('/api/news', methods=['GET'])
def get_news():
    try:
        cache_key = "news"
        now = time.time()
        with _news_cache_lock:
            if cache_key in _news_cache:
                expire, data = _news_cache[cache_key]
                if now < expire:
                    return jsonify(data)

        sources = [
            ("BBC", "http://feeds.bbci.co.uk/news/rss.xml"),
            ("NYTimes", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
            ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml")
        ]

        articles = []

        for source_name, url in sources:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            }

            response = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(response.content)

            for entry in feed.entries:
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        dt = datetime(*entry.published_parsed[:6])
                        published = dt.isoformat()
                    else:
                        published = ""
            
                    title = clean_html(entry.get('title', ''))
                    description = clean_html(entry.get('summary', ''))
            
                    image = None
                    if entry.get("media_content"):
                        image = entry.media_content[0].get("url")
            
                    pred = hybrid_model.predict(title, description, source_url=entry.get("link", ""))
            
                    articles.append({
                        "title": title,
                        "description": description,
                        "url": entry.get("link", ""),
                        "urlToImage": image,
                        "source": source_name,
                        "publishedAt": published,
                        "verdict": pred['verdict'],
                        "confidence": pred['confidence']
                    })
            
                except Exception as e:
                    print("RSS parse error:", e)
                    continue

        articles.sort(key=lambda x: x['publishedAt'], reverse=True)

        response = {
            'status': 'ok',
            'totalResults': len(articles),
            'articles': articles[:12]
        }

        with _news_cache_lock:
            _news_cache[cache_key] = (now + _NEWS_CACHE_TTL, response)

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
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
    'createdAt': datetime.utcnow(),
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

    created_at_iso = None
    try:
        if 'createdAt' in user and user['createdAt'] is not None:
            created_at_iso = user['createdAt'].isoformat()
    except Exception:
        created_at_iso = None

    return jsonify({
        'valid': True,
        'user': {
            'email': user['email'],
            'userId': str(user['_id']),
            'joinDate': created_at_iso
        },
        'isAdmin': user.get('isAdmin', False)
    }), 200

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

    if url and not headline and not content:
        try:
            headline, content = extract_url_content(url)
        except Exception as e:
            logger.warning("Failed to extract URL content for %s: %s", url, e)
            return jsonify({'error': 'Could not read that URL. Please paste the headline and content instead.'}), 400

    if not headline and not content:
        return jsonify({'error': 'No content provided'}), 400

    prediction, community = predict_with_community_feedback(headline, content, source_url=url)

    try:
        eng = EngineeredFeatures()
        feat_arr = eng.transform([(headline, content)])[0]

        feat_names = [
            'sensationalism','source_credibility','specificity','emotional_language',
            'passive_voice','quote_presence','expert_reference','controversy',
            'multi_source','coherence'
        ]

        factors = dict(zip(feat_names, [round(float(x), 3) for x in feat_arr]))

        top_factors = sorted(factors.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    except Exception:
        factors = {}
        top_factors = []

    analysis = {
        'userId': user_obj_id,
        'headline': headline,
        'content': content,
        'url': url,
        'category': category,
        'verdict': prediction['verdict'],
        'confidence': prediction['confidence'],
        'mlScore': prediction.get('mlScore', 50),
        'modelScore': prediction.get('modelScore'),
        'heuristicScore': prediction.get('heuristicScore'),
        'bertScore': prediction.get('bertScore'),
        'hasEvidence': prediction.get('hasEvidence'),
        'fakeRisk': prediction.get('fakeRisk'),
        'articleLike': prediction.get('articleLike'),
        'trustedSource': prediction.get('trustedSource'),
        'factors': factors,
        'topFactors': top_factors,
        'communityVote': {'real': 0, 'fake': 0, 'uncertain': 0},
        'communityInfluence': prediction.get('communityInfluence'),
        'createdAt': datetime.utcnow()
    }

    result = analyses_collection.insert_one(analysis)

    users_collection.update_one(
        {'_id': user_obj_id},
        {'$inc': {'analysisCount': 1}}
    )
    reasoning = []

    if prediction.get("modelScore", 50) > prediction.get("heuristicScore", 50):
        reasoning.append("ML model confidence is stronger than heuristic patterns.")
    else:
        reasoning.append("Heuristic signals influenced the decision more strongly.")

    if prediction.get("agreement", 0) > 25:
        reasoning.append("Model disagreement detected → lower reliability.")

    if prediction.get("trustedSource"):
        reasoning.append("Source is verified → credibility boosted.")

    if prediction.get("fakeRisk"):
        reasoning.append("High fake-news linguistic patterns detected.")

    claims = extract_claims(content)
    claim_scores = []
    for c in claims[:5]:
        try:
            pred = hybrid_model.predict("", c)
            claim_scores.append(pred.get('mlScore', 50))
        except Exception:
            claim_scores.append(50)

    claim_variance = np.std(claim_scores) if claim_scores else 0

    return jsonify({
        'analysisId': str(result.inserted_id),
        'headline': headline,
        'verdict': prediction['verdict'],
        'confidence': prediction['confidence'],
        'mlScore': prediction.get('mlScore', 50),
        'modelScore': prediction.get('modelScore'),
        "hasEvidence": bool(re.search(r"(source|report|according to|study)", content.lower())),
        'heuristicScore': prediction.get('heuristicScore'),
        'bertScore': prediction.get('bertScore'),
        'hasEvidence': prediction.get('hasEvidence'),
        'fakeRisk': prediction.get('fakeRisk'),
        'articleLike': prediction.get('articleLike'),
        'trustedSource': prediction.get('trustedSource'),
        'factors': factors,
        'topFactors': top_factors,
        'communityConsensus': community,
        'createdAt': analysis['createdAt'].isoformat(),
        'reasoning': reasoning,
        'claimConsistency': round(100 - claim_variance, 2)
    }), 201
from operator import itemgetter

def extract_claims(text):
    sentences = re.split(r'[.!?]', text)
    return [s.strip() for s in sentences if len(s.split()) > 6]

@app.route('/api/analyze/explain', methods=['POST'])
@jwt_required()
def explain_analysis():
    data = request.get_json() or {}
    headline = data.get('headline', '')
    content = data.get('content', '')

    pred = hybrid_model.predict(headline, content)

    eng = EngineeredFeatures()
    feat_arr = eng.transform([(headline, content)])[0]

    feat_names = [
        'sensationalism','source_credibility','specificity','emotional_language',
        'passive_voice','quote_presence','expert_reference','controversy',
        'multi_source','coherence'
    ]

    factors = dict(zip(feat_names, [float(round(float(x), 3)) for x in feat_arr]))
    
    factors = pred.get('factors') or {}

    try:
        eng = EngineeredFeatures()
        feat_arr = eng.transform([(headline, content)])[0]
        feat_names = ['sensationalism','source_cred','specificity','emotional','passive','quote_factor','expert_ref','controversy','multi_source','coherence']
        factors = dict(zip(feat_names, [float(round(float(x), 3)) for x in feat_arr]))
    except Exception:
        pass

    top3 = sorted(factors.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
    return jsonify({'factors': factors, 'topContributors': top3, 'rawPrediction': pred}), 200

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
            if not analyses_collection.find_one({'_id': analysis_obj_id}):
                print(f"[WARN] create_review: analysis id provided but not found: {analysis_id}")
                analysis_obj_id = None
        except Exception:
            print(f"[WARN] create_review: invalid analysis id format: {analysis_id}")
            analysis_obj_id = None

    review_doc = {
        'userId': user_obj_id,
        'analysisId': analysis_obj_id,   
        'verdict': verdict,
        'text': text,
        'helpful': 0,
        'createdAt': datetime.utcnow()
    }

    try:
        result = reviews_collection.insert_one(review_doc)
        users_collection.update_one({'_id': user_obj_id}, {'$inc': {'reviewCount': 1}})
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
            'modelScore': a.get('modelScore'),
            'heuristicScore': a.get('heuristicScore'),
            'bertScore': a.get('bertScore'),
            'hasEvidence': a.get('hasEvidence'),
            'fakeRisk': a.get('fakeRisk'),
            'articleLike': a.get('articleLike'),
            'trustedSource': a.get('trustedSource'),
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error'}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
