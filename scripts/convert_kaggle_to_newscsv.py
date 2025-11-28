#!/usr/bin/env python3
"""
convert_kaggle_to_newscsv.py

Usage:
  python convert_kaggle_to_newscsv.py --fake data/Fake.csv --true data/True.csv --out data/news_dataset.csv --shuffle --seed 42 --limit 50000

Produces a CSV with columns: headline,content,label
label: 1 = real, 0 = fake
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys
import html
import re

def load_csv_try(path):
    # try different encodings to be safe
    for enc in ('utf-8','utf-8-sig','latin1','iso-8859-1'):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise RuntimeError(f"Failed to read CSV: {path}")

def clean_text(s):
    if pd.isna(s):
        return ''
    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s)
    # remove excessive whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def pick_text_columns(df):
    # heuristics: prefer ('title' or 'headline') for headline; ('text' or 'content' or 'article') for content
    cols = [c.lower() for c in df.columns]
    headline_col = None
    content_col = None
    for candidate in ('title','headline','head','subject'):
        if candidate in cols:
            headline_col = df.columns[cols.index(candidate)]
            break
    for candidate in ('text','content','article','body'):
        if candidate in cols:
            content_col = df.columns[cols.index(candidate)]
            break
    # fallback to first two string columns
    if headline_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                headline_col = c
                break
    if content_col is None:
        for c in df.columns:
            if df[c].dtype == object and c != headline_col:
                content_col = c
                break
    return headline_col, content_col

def main(args):
    fake_path = args.fake
    true_path = args.true
    out_path = args.out

    if not os.path.exists(fake_path):
        print("Fake CSV not found:", fake_path)
        sys.exit(1)
    if not os.path.exists(true_path):
        print("True CSV not found:", true_path)
        sys.exit(1)

    print("Loading Fake CSV:", fake_path)
    df_fake = load_csv_try(fake_path)
    print("Loading True CSV:", true_path)
    df_true = load_csv_try(true_path)

    h_fake, c_fake = pick_text_columns(df_fake)
    h_true, c_true = pick_text_columns(df_true)

    if not h_fake or not c_fake or not h_true or not c_true:
        print("Failed to detect headline/content columns. Inspect input CSV headers.")
        print("Fake columns:", df_fake.columns.tolist())
        print("True columns:", df_true.columns.tolist())
        sys.exit(1)

    print("Detected columns - Fake:", h_fake, c_fake, " | True:", h_true, c_true)

    df_f = pd.DataFrame({
        'headline': df_fake[h_fake].astype(str).apply(clean_text),
        'content': df_fake[c_fake].astype(str).apply(clean_text),
        'label': 0
    })

    df_t = pd.DataFrame({
        'headline': df_true[h_true].astype(str).apply(clean_text),
        'content': df_true[c_true].astype(str).apply(clean_text),
        'label': 1
    })

    df = pd.concat([df_f, df_t], ignore_index=True)

    if args.limit and args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed if args.seed else None)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed if args.seed else None).reset_index(drop=True)

    # ensure directory exists
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8')
    print("Saved merged dataset to:", out_path)
    print("Rows:", len(df))
    print("Sample:")
    print(df.head(5).to_dict(orient='records'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake', required=True, help='Path to Fake.csv')
    parser.add_argument('--true', required=True, help='Path to True.csv')
    parser.add_argument('--out', required=True, help='Output CSV path (e.g., data/news_dataset.csv)')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle rows')
    parser.add_argument('--limit', type=int, default=0, help='Limit total row count (0 = all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffle/limit')
    args = parser.parse_args()
    main(args)
