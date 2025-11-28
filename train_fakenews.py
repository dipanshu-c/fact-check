# train_fakenews.py
import argparse
import os
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from scipy.sparse import hstack

from models.features import EngineeredFeatures  # keep your existing engineered features class

# ---------- helpers ----------
def make_article_key(headline, content, url=None, analysisId=None):
    if analysisId:
        return f"id::{str(analysisId)}"
    if url:
        return f"url::{url}"
    concat = (str(headline) + "||" + str(content)).encode('utf-8')
    return "hash::" + hashlib.sha1(concat).hexdigest()

def aggregate_reviews_from_df(reviews_df):
    groups = defaultdict(list)
    for _, row in reviews_df.iterrows():
        aid = row.get('analysisId') if 'analysisId' in row else None
        # ensure string form for ObjectId values
        if pd.notna(aid) and not isinstance(aid, str):
            aid = str(aid)
        url = row.get('url') if 'url' in row else None
        headline = row.get('headline') if 'headline' in row else ''
        content = row.get('content') if 'content' in row else ''
        verdict = (row.get('verdict') or '').strip().lower()
        if verdict in ('true','real'):
            verdict_norm = 'real'
        elif verdict in ('false','fake'):
            verdict_norm = 'fake'
        elif verdict == 'uncertain':
            verdict_norm = 'uncertain'
        else:
            continue
        key = make_article_key(headline, content, url=url, analysisId=aid)
        groups[key].append(verdict_norm)
    agg = {}
    for key, votes in groups.items():
        counts = Counter(votes)
        if (counts.get('real',0) + counts.get('fake',0)) == 0:
            consensus = 'uncertain'
        else:
            if counts.get('real',0) > counts.get('fake',0):
                consensus = 'real'
            elif counts.get('fake',0) > counts.get('real',0):
                consensus = 'fake'
            else:
                consensus = 'uncertain'
        n = len(votes)
        support = counts.get(consensus, 0)
        agg[key] = {
            'consensus': consensus,
            'counts': dict(counts),
            'n_reviews': n,
            'support_ratio': support / n if n > 0 else 0.0
        }
    return agg

# ---------- load reviews (Mongo) ----------
def load_reviews_from_mongo(mongo_uri, mongo_db, mongo_collection):
    from pymongo import MongoClient
    client = MongoClient(mongo_uri)
    coll = client[mongo_db][mongo_collection]
    docs = list(coll.find({}))
    rows = []
    for d in docs:
        rows.append({
            'analysisId': str(d.get('analysisId')) if d.get('analysisId') else (str(d.get('_id')) if d.get('_id') else None),
            'verdict': d.get('verdict'),
            'headline': d.get('headline'),
            'url': d.get('url'),
            'content': d.get('content'),
            'username': d.get('username')
        })
    return pd.DataFrame(rows)

# ---------- main training pipeline ----------
def build_and_train(df, headline_col, text_col, label_col, out_path,
                    test_size=0.2, random_state=42, clf_name='logreg',
                    reviews_agg=None, merge_strategy='override'):
    X_head = df[headline_col].fillna('').astype(str)
    X_text = df[text_col].fillna('').astype(str)
    y_raw = df[label_col].astype(str).str.strip().str.lower()

    def normalize_label(lbl):
        if lbl in ('1','true','real','truth','genuine'):
            return 'real'
        if lbl in ('0','false','fake','untrue'):
            return 'fake'
        return lbl

    y_norm = y_raw.apply(normalize_label)

    # keys for matching with reviews
    article_keys = [make_article_key(h, t) for h, t in zip(X_head.tolist(), X_text.tolist())]

    sample_weights = np.ones(len(df), dtype=float)

    if reviews_agg:
        for i, key in enumerate(article_keys):
            if key in reviews_agg:
                r = reviews_agg[key]
                cons = r['consensus']
                support_ratio = r['support_ratio']
                nrev = r['n_reviews']
                if merge_strategy == 'override' and cons in ('real','fake'):
                    y_norm.iat[i] = cons
                elif merge_strategy == 'weight':
                    # scale weight with support_ratio and log(nrev+1)
                    sample_weights[i] = 1.0 + support_ratio * np.log1p(nrev) * 2.0

    le = LabelEncoder()
    y_enc = le.fit_transform(y_norm)

    X_combined = (X_head + " " + X_text).values
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')
    X_tfidf = tfidf.fit_transform(X_combined)

    eng = EngineeredFeatures()
    X_eng = eng.transform(list(zip(X_head.tolist(), X_text.tolist())))

    scaler = StandardScaler()
    X_eng_scaled = scaler.fit_transform(X_eng)

    X_full = hstack([X_tfidf, X_eng_scaled])

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X_full, y_enc, sample_weights, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    if clf_name == 'logreg':
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')
    else:
        clf = LogisticRegression(max_iter=2000)

    print("Training classifier...")
    if np.any(sw_train != 1.0):
        clf.fit(X_train, y_train, sample_weight=sw_train)
    else:
        clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            print("ROC AUC:", roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    artifact = {
        'tfidf': tfidf,
        'eng': eng,
        'scaler': scaler,
        'clf': clf,
        'label_encoder': le
    }
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(artifact, f)
    print("Saved model to", out_path)
    return artifact

# ---------- CLI ----------
def load_df(path, text_col_candidates=None, headline_col_candidates=None, label_col_candidates=None):
    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    cols = set(df.columns.str.lower())
    text_col = None
    headline_col = None
    label_col = None

    text_candidates = text_col_candidates or ['text','content','article','body','originaltweet']
    headline_candidates = headline_col_candidates or ['title','headline','head','summary']
    label_candidates = label_col_candidates or ['label','class','target','sentiment','truth']

    for c in df.columns:
        lc = c.lower()
        if not text_col and lc in text_candidates:
            text_col = c
        if not headline_col and lc in headline_candidates:
            headline_col = c
        if not label_col and lc in label_candidates:
            label_col = c

    if not headline_col:
        df['__headline__'] = ''
        headline_col = '__headline__'

    if not text_col:
        possible = [c for c in df.columns if c not in [headline_col, label_col]]
        text_col = possible[0] if possible else headline_col

    if not label_col:
        raise ValueError("Could not find label column. Provide --label_col explicitly.")

    return df, headline_col, text_col, label_col

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV file with news')
    parser.add_argument('--headline_col', default=None)
    parser.add_argument('--content_col', default=None)
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--out', default='models/fakenews_model.pkl')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--clf', default='logreg', choices=['logreg'])
    parser.add_argument('--reviews_mongo', action='store_true', help='Fetch reviews from MongoDB')
    parser.add_argument('--mongo_uri', default=os.getenv('MONGO_URI', 'mongodb://localhost:27017/fakenews_detector'))
    parser.add_argument('--mongo_db', default=os.getenv('MONGO_DB', 'fakenews_detector'))
    parser.add_argument('--mongo_collection', default=os.getenv('REVIEWS_COLLECTION', 'reviews'))
    parser.add_argument('--review_merge_strategy', choices=['override','weight'], default='override')
    args = parser.parse_args()

    df, hc, tc, lc = load_df(args.data,
                             headline_col_candidates=[args.headline_col] if args.headline_col else None,
                             text_col_candidates=[args.content_col] if args.content_col else None,
                             label_col_candidates=[args.label_col] if args.label_col else None)

    print(f"[train] Using headline_col={hc}, text_col={tc}, label_col={lc}, n={len(df)}")

    reviews_agg = None
    if args.reviews_mongo:
        print("Loading reviews from MongoDB:", args.mongo_uri, args.mongo_db, args.mongo_collection)
        rev_df = load_reviews_from_mongo(args.mongo_uri, args.mongo_db, args.mongo_collection)
        reviews_agg = aggregate_reviews_from_df(rev_df)
        print("Aggregated", len(reviews_agg), "distinct reviewed articles")

    build_and_train(df, hc, tc, lc, args.out, test_size=args.test_size, random_state=args.seed,
                    clf_name=args.clf, reviews_agg=reviews_agg, merge_strategy=args.review_merge_strategy)

if __name__ == '__main__':
    main()
