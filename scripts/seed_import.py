"""
Romberger: seed data import

Loads our existing labeled feature dataset into Supabase as `samples' rows so the retraining pipeline has historical data to train on.

Uses:
    pip install supabase pandas
    export SB_URL="https://xmxyschvrqcqhqznjdgu.supabase.co"
    export SB_SERVICE_ROLE_KEY="..."
    python scripts/seed_import.py path/to/dataset.csv
"""

import os
import sys
import uuid
from datetime import datetime, timezone

import pandas as pd
from supabase import create_client


SUPABASE_URL = os.environ['SB_URL']
SUPABASE_SERVICE_ROLE_KEY = os.environ['SB_SERVICE_ROLE_KEY']

FEATURE_COLUMNS = ['mean', 'median', 'std', 'skewness', 'kurtosis', 'path_length']


def main():
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows from {csv_path}')

    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing:
      sys.exit(f'ERROR: CSV is missing columns: {missing}')

    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for idx, row in df.iterrows():
        # One session_id per subject
        session_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f'subject-{row["subject_id"]}'))
        sample_id  = str(uuid.uuid5(uuid.NAMESPACE_OID, f'seed-{idx}'))

        rows.append({
            'sample_id':          sample_id,
            'timestamp':          timestamp,
            'subject_session_id': session_id,
            'label':              row['label'],
            'extracted_features': {f: float(row[f]) for f in FEATURE_COLUMNS},
            'prediction_output':  row['label'],  
            'confidence':         1.0,
            'model_version':      'seed_v1',
            'duration_seconds':   30.0,
            'sample_rate':        100,
            'n_samples':          3000,
            'storage_path':       f'seed/{sample_id}.csv',
        })


    print(f'Inserting {len(rows)} rows...')
    client.table('samples').upsert(rows, on_conflict='sample_id').execute()
    print('Done.')


if __name__ == '__main__':
    main()
