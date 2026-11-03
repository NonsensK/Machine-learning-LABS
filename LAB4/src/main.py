"""Lab 4: deterministic, leakage-free Adult dataset pipeline."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent.parent
CATEGORICAL = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
NUMERIC = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
FEATURES = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
RANGES = {'age':(17,90),'hours-per-week':(1,99),'capital-gain':(0,99999),'capital-loss':(0,4356)}

class TrainQuantileClipper(BaseEstimator, TransformerMixin):
    """Compute per-column bounds using training rows only."""
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(arr, self.lower, axis=0)
        self.high_ = np.nanquantile(arr, self.upper, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(len(self.low_))], dtype=object)
        return np.asarray(input_features, dtype=object)

def add_features(X):
    X = X.copy()
    X['capital_net'] = pd.to_numeric(X['capital-gain'],errors='coerce') - pd.to_numeric(X['capital-loss'],errors='coerce')
    # Adult age >=17; denominator safe for valid input. Invalid ages become NaN and get imputed.
    denom = pd.to_numeric(X['age'],errors='coerce') - 16
    X['hours_per_age'] = pd.to_numeric(X['hours-per-week'],errors='coerce') / denom.where(denom > 0)
    return X

def make_preprocessor(engineered):
    num = NUMERIC + (['capital_net','hours_per_age'] if engineered else [])
    numeric_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='median')),
        ('clipper',TrainQuantileClipper()),
        ('scaler',StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([('numeric',numeric_pipe,num),('categorical',cat_pipe,CATEGORICAL)])

def validation(df):
    result = {'row_count':int(len(df)), 'column_count':int(len(df.columns)), 'expected_rows':48842,'expected_columns':15,
              'schema_violations':{},'range_violations':{},'missing_counts':{},'duplicate_full_rows':int(df.duplicated().sum())}
    for col in FEATURES + ['income']:
        if col not in df:
            result['schema_violations'][col] = 'MISSING COLUMN'
            continue
        s = df[col]
        result['missing_counts'][col] = int(s.isna().sum() + s.astype('string').str.strip().eq('?').fillna(False).sum())
        if col in NUMERIC:
            parsed = pd.to_numeric(s,errors='coerce')
            result['schema_violations'][col] = int((parsed.isna() & s.notna() & ~s.astype('string').str.strip().eq('?').fillna(False)).sum())
            if col in RANGES:
                lo,hi=RANGES[col]
                result['range_violations'][col]=int((parsed.notna() & ~parsed.between(lo,hi)).sum())
        else:
            result['schema_violations'][col] = int((s.notna() & ~s.map(lambda x: isinstance(x, str))).sum()) if not isinstance(s.dtype, pd.CategoricalDtype) else 0
    result['schema_violations']['unexpected_columns'] = sorted(set(df.columns)-set(FEATURES)-{'income'})
    result['row_count_matches'] = len(df)==48842
    result['column_count_matches'] = len(df.columns)==15
    return result

def clean_tokens(df):
    out=df.copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype('string').str.strip().replace('?',pd.NA).astype(object)
        out[c] = out[c].where(pd.notna(out[c]),np.nan)
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c],errors='coerce')
    return out

def label_audit(df, seed):
    # Duplicates are grouped by all 14 raw features; target excluded.
    frame = df[FEATURES+['income']].copy()
    frame[FEATURES] = frame[FEATURES].astype('string').fillna('<MISSING>')
    grouped = frame.groupby(FEATURES,dropna=False,observed=True)['income'].agg(['nunique','size'])
    conflicts = grouped[grouped['nunique']>1]
    sample = df.sample(n=min(50,len(df)),random_state=seed)
    manual_checks = []
    for idx,row in sample.iterrows():
        failures=[]
        for c,(lo,hi) in RANGES.items():
            v=pd.to_numeric(row[c],errors='coerce')
            if pd.isna(v) or not lo<=v<=hi:
                failures.append(c)
        for c in NUMERIC:
            if pd.isna(pd.to_numeric(row[c],errors='coerce')):
                failures.append(f'{c}:non-numeric')
        if str(row['income']).strip() not in {'<=50K','>50K'}:
            failures.append('income:unexpected-label')
        manual_checks.append({'source_index':str(idx),'impossible_value_flags':failures})
    return {'sample_size':len(sample),'sample_seed':seed,'sampled_rows_flagged':sum(bool(x['impossible_value_flags']) for x in manual_checks),
            'duplicate_feature_groups_with_conflicting_labels':int(len(conflicts)),
            'rows_in_conflicting_groups':int(conflicts['size'].sum()) if len(conflicts) else 0,
            'sample_review':manual_checks,
            'note':'Automated dictionary/range review of 50 sampled rows; human manual verification remains required.'}

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run(config, out, csv_path=None):
    out.mkdir(parents=True,exist_ok=True)
    seed=int(config['seed'])
    if csv_path:
        raw=pd.read_csv(csv_path,na_values=[],keep_default_na=False)
        source=f'local CSV: {csv_path}'
    else:
        adult=fetch_openml(name='adult',version=2,as_frame=True)
        raw=adult.frame.copy(deep=True)
        source='OpenML adult version 2 (UCI Adult)'
    # Preserve untouched raw input: ingestion and validation do not mutate raw.
    if 'class' in raw.columns and 'income' not in raw.columns:
        raw=raw.rename(columns={'class':'income'})
    if 'income' not in raw.columns:
        raise ValueError('Expected income target column')
    report=validation(raw)
    if set(FEATURES+['income']) - set(raw.columns):
        raise ValueError('Required columns missing; see dataset schema')
    raw=raw[FEATURES+['income']].copy()
    (out/'validation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    labels=raw['income'].astype('string').str.strip().str.replace(r'\.$','',regex=True)
    if not labels.isin(['<=50K','>50K']).all():
        raise ValueError('Unexpected or missing income label; see validation report')
    y=(labels=='>50K').astype(int)
    X=raw[FEATURES].copy()
    X_train_raw,X_test_raw,y_train,y_test=train_test_split(X,y,test_size=float(config['test_size']),stratify=y,random_state=seed)
    # Duplicates are removed ONLY in training; test rows and ordering are preserved.
    keep=~X_train_raw.duplicated(keep='first')
    removed=int((~keep).sum())
    X_train_raw=X_train_raw.loc[keep].copy()
    y_train=y_train.loc[keep].copy()
    X_train=clean_tokens(X_train_raw)
    X_test=clean_tokens(X_test_raw)
    audit_frame=raw.copy()
    audit_frame['income']=labels
    audit=label_audit(audit_frame,seed)
    audit_frame.sample(n=min(50,len(audit_frame)),random_state=seed).to_csv(out/'label_review_50.csv',index=True)
    (out/'label_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    scores={}
    fitted={}
    for engineered,name in [(False,'baseline'),(True,'engineered')]:
        Xt=add_features(X_train) if engineered else X_train
        Xv=add_features(X_test) if engineered else X_test
        pre=make_preprocessor(engineered)
        model=Pipeline([('preprocessor',pre),('classifier',LogisticRegression(max_iter=2000,random_state=seed))])
        model.fit(Xt,y_train)
        pred=model.predict(Xv)
        scores[name]={'accuracy':float(accuracy_score(y_test,pred)),'f1':float(f1_score(y_test,pred)),
                      'train_rows_after_dedup':len(X_train),'test_rows':len(X_test)}
        fitted[name]=model
    joblib.dump(fitted['engineered'].named_steps['preprocessor'],out/'preprocessor.joblib')
    joblib.dump(fitted['engineered'],out/'model.joblib')
    # Hash canonical processed data with explicit stable row order, column order, and float formatting.
    processed=fitted['engineered'].named_steps['preprocessor'].transform(add_features(X_train))
    if hasattr(processed,'toarray'): processed=processed.toarray()
    feature_names=fitted['engineered'].named_steps['preprocessor'].get_feature_names_out()
    processed_df=pd.DataFrame(processed,columns=feature_names)
    processed_df.insert(0,'income',y_train.to_numpy())
    processed_path=out/'processed_train.csv'
    processed_df.to_csv(processed_path,index=False,float_format='%.12g',lineterminator='\n')
    hashval=sha256_file(processed_path)
    (out/'processed_train.sha256').write_text(f'{hashval}  processed_train.csv\n',encoding='utf-8')
    report.update({'train_before_dedup':len(keep),'train_duplicates_removed':removed,'train_after_dedup':len(X_train),
                   'test_rows_untouched':len(X_test),'train_test_split_before_cleaning':True})
    (out/'validation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    metadata={'dataset_source':source,'seed':seed,'test_size':float(config['test_size']),
              'missing_token':config.get('missing_token','?'), 'clip_quantiles':config.get('clip_quantiles',[0.01,0.99]),
              'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,
              'processed_train_sha256':hashval,'hash_file':'processed_train.csv'}
    (out/'run_metadata.yaml').write_text(yaml.safe_dump(metadata,sort_keys=True),encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7,4))
    x=np.arange(2); width=.35
    ax.bar(x-width/2,[scores[k]['accuracy'] for k in ('baseline','engineered')],width,label='Accuracy')
    ax.bar(x+width/2,[scores[k]['f1'] for k in ('baseline','engineered')],width,label='F1')
    ax.set_xticks(x,['Baseline','Engineered']);ax.set_ylim(0,1);ax.legend();ax.set_ylabel('Score')
    fig.tight_layout();fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'metrics':scores,'sha256':hashval,'results':str(out)},indent=2))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config.yaml'))
    parser.add_argument('--output',default=str(ROOT/'results'))
    parser.add_argument('--csv',default=None,help='Optional local Adult CSV with 15 named columns for offline use')
    args=parser.parse_args()
    config_path=Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f'Configuration file not found: {config_path}')
    with open(config_path,encoding='utf-8') as f:config=yaml.safe_load(f)
    run(config,Path(args.output),args.csv)
if __name__=='__main__':main()
