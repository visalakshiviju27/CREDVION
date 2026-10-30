"""CREDVION reusable, leakage-safe credit risk auditing engine."""
import io, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score

TARGET='default payment next month'
PROTECTED=['SEX','AGE','EDUCATION','MARRIAGE']

def read_table(blob,filename):
    name=filename.lower()
    if name.endswith('.csv'):
        df=pd.read_csv(io.BytesIO(blob))
    elif name.endswith('.xls'):
        df=pd.read_excel(io.BytesIO(blob),engine='xlrd')
    elif name.endswith('.xlsx'):
        df=pd.read_excel(io.BytesIO(blob),engine='openpyxl')
    else: raise ValueError('Upload CSV, XLS, or XLSX only.')
    df.columns=[str(x).strip() for x in df.columns]
    if TARGET not in df.columns and len(df)>1 and name.endswith(('.xls','.xlsx')):
        # UCI default-of-credit-card-clients XLS has a title row before its header.
        if str(df.columns[0]).startswith('Unnamed') or 'X1' in df.columns:
            df=pd.read_excel(io.BytesIO(blob),header=1,engine='xlrd' if name.endswith('.xls') else 'openpyxl')
            df.columns=[str(x).strip() for x in df.columns]
    if df.empty: raise ValueError('The dataset has no rows.')
    if len(df)>200000:raise ValueError('Maximum 200,000 rows per upload for the demo.')
    if len(df.columns)>150:raise ValueError('Maximum 150 columns per upload.')
    return df

def safe_number(v):
    if v is None:return None
    try:
        n=float(v)
        return round(n,5) if math.isfinite(n) else None
    except (TypeError,ValueError):return None

def evaluate(y,p,scores=None):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    if not len(y):raise ValueError('No predictions to evaluate.')
    tn,fp,fn,tp=[int(v) for v in confusion_matrix(y,p,labels=[0,1]).ravel()]
    result={'count':len(y),'correct':tn+tp,'incorrect':fp+fn,'accuracy':safe_number((tn+tp)/len(y)),
        'error_rate':safe_number((fp+fn)/len(y)), 'precision':safe_number(precision_score(y,p,zero_division=0)),
        'recall':safe_number(recall_score(y,p,zero_division=0)),'f1':safe_number(f1_score(y,p,zero_division=0)),
        'true_negatives':tn,'false_positives':fp,'false_negatives':fn,'true_positives':tp,
        'false_positive_rate':safe_number(fp/(fp+tn)) if fp+tn else None,
        'false_negative_rate':safe_number(fn/(fn+tp)) if fn+tp else None,
        'predicted_high_risk_rate':safe_number(np.mean(p)), 'actual_default_rate':safe_number(np.mean(y)), 'roc_auc':None}
    if scores is not None and len(set(y))==2:
        result['roc_auc']=safe_number(roc_auc_score(y,scores))
    return result

def group_audit(y,p,df,group_cols):
    out={};y=np.asarray(y);p=np.asarray(p)
    for col in group_cols:
        if col not in df:continue
        s=df[col].reset_index(drop=True)
        if col.upper()=='AGE' and pd.api.types.is_numeric_dtype(s):
            s=pd.cut(pd.to_numeric(s,errors='coerce'),[0,25,35,45,55,150],labels=['≤25','26–35','36–45','46–55','56+']).astype(str)
        if col.upper()=='SEX':s=s.map({1:'Male (1)',2:'Female (2)'}).fillna(s.astype(str))
        s=s.fillna('Missing').astype(str)
        if s.nunique()>30:continue
        rows=[]
        for name in sorted(s.unique()):
            mask=(s==name).to_numpy()
            r=evaluate(y[mask],p[mask]);r['group']=str(name);rows.append(r)
        valid=lambda key:[r[key] for r in rows if r[key] is not None]
        gap=lambda key:safe_number(max(valid(key))-min(valid(key))) if len(valid(key))>=2 else None
        out[col]={'rows':rows,'accuracy_gap':gap('accuracy'),'fpr_gap':gap('false_positive_rate'),
                  'fnr_gap':gap('false_negative_rate'),'demographic_parity_difference':gap('predicted_high_risk_rate'),
                  'equal_opportunity_difference':gap('false_negative_rate'),
                  'equalized_odds_max_gap':max([v for v in [gap('false_positive_rate'),gap('false_negative_rate')] if v is not None],default=None)}
    return out

def preprocess(df,target):
    if target not in df:raise ValueError('Target column not found.')
    source=len(df);missing=int(df.isna().sum().sum());duplicates=int(df.duplicated().sum())
    df=df.drop_duplicates().copy()
    df=df.dropna(subset=[target])
    vals=pd.to_numeric(df[target],errors='coerce')
    if not vals.dropna().isin([0,1]).all():raise ValueError('Target must be encoded as 0/1. Map other labels before training.')
    df=df.loc[vals.notna()].copy();df[target]=vals.loc[vals.notna()].astype(int)
    if df[target].nunique()!=2:raise ValueError('Training requires both 0 and 1 outcomes.')
    if df[target].value_counts().min()<4:raise ValueError('At least four records of each outcome are required.')
    return df,{'original_rows':source,'rows_after_cleaning':len(df),'removed_rows':source-len(df),
               'duplicate_rows':duplicates,'missing_cells_before':missing,'missing_cells_after':int(df.isna().sum().sum())}

def train_three(df,target,demographics=None):
    df,clean=preprocess(df,target)
    demographics=[c for c in (demographics or PROTECTED) if c in df and c!=target]
    excluded=[target]+demographics+[c for c in df if c.strip().lower() in ['id','customer_id','customer id']]
    X=df.drop(columns=excluded)
    if not len(X.columns):raise ValueError('No predictive features remain after excluding target and demographic columns.')
    y=df[target]
    train_idx,test_idx=train_test_split(np.arange(len(df)),test_size=.25,stratify=y,random_state=42)
    nums=X.select_dtypes(include='number').columns.tolist();cats=[c for c in X if c not in nums]
    transforms=[]
    if nums:transforms.append(('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums))
    if cats:transforms.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
    models={'Credit AI – A (Logistic Regression)':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),
            'Credit AI – B (Random Forest)':RandomForestClassifier(n_estimators=90,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1),
            'Credit AI – C (Gradient Boosting)':GradientBoostingClassifier(n_estimators=90,random_state=42)}
    results={};details={};ytest=y.iloc[test_idx].to_numpy()
    for name,clf in models.items():
        pipe=Pipeline([('prep',ColumnTransformer(transforms)),('model',clf)])
        pipe.fit(X.iloc[train_idx],y.iloc[train_idx]);scores=pipe.predict_proba(X.iloc[test_idx])[:,1]
        pred=(scores>=.5).astype(int)
        metrics=evaluate(ytest,pred,scores)
        audit=group_audit(ytest,pred,df.iloc[test_idx].reset_index(drop=True),demographics)
        results[name]={'metrics':metrics,'fairness':audit}
        details[name]={'scores':scores,'actual':ytest,'groups':df.iloc[test_idx].reset_index(drop=True), 'pipeline':pipe}
    return {'dataset_rows':len(df),'train_rows':len(train_idx),'test_rows':len(test_idx),'cleaning':clean,
            'target':target,'demographics':demographics,'models':results,
            'method':'Stratified 75/25 holdout; random seed 42; preprocessing fitted only on training data; demographic columns excluded from model features.',
            'caveats':['Demo models, not proprietary bank systems.','Credit-card default risk is not the same as a loan approval decision.',
                       'Observed disparities do not establish unlawful discrimination.','Small groups may yield unstable rates.']},details

def audit_external(df,actual,prediction,demographics=None,positive_label='1',probability=None):
    for col in [actual,prediction]:
        if col not in df:raise ValueError(f'Missing column: {col}')
    if actual==prediction:raise ValueError('Actual and predicted columns must differ.')
    positive=str(positive_label).strip().lower()
    def encode(s):
        clean=s.astype(str).str.strip().str.lower()
        # Accept 0/1, plus user-chosen positive class. Reject ambiguous extra labels.
        positive_mask=clean.eq(positive)
        unique=set(clean.unique());allowed={positive,'0','1'}
        if not unique.issubset(allowed):raise ValueError('Actual and prediction labels must be binary (0/1), with the selected positive label.')
        return positive_mask.astype(int).to_numpy()
    keep=df[actual].notna() & df[prediction].notna()
    if not keep.any():raise ValueError('No rows with both actual and predicted outcomes.')
    subset=df.loc[keep].reset_index(drop=True)
    y=encode(subset[actual]);p=encode(subset[prediction])
    scores=None
    if probability and probability in subset:
        scores=pd.to_numeric(subset[probability],errors='coerce').to_numpy()
        if np.isnan(scores).any() or np.any((scores<0)|(scores>1)):
            raise ValueError('Probability scores must be between 0 and 1, with no missing values.')
    groups=[c for c in (demographics or []) if c in subset and c not in [actual,prediction]]
    return {'dataset_rows':len(df),'evaluated_rows':len(subset),'excluded_rows':len(df)-len(subset),
            'metrics':evaluate(y,p,scores),'fairness':group_audit(y,p,subset,groups),
            'method':'External predictions compared with uploaded ground-truth labels; no external model executed or retrained.',
            'caveats':['Only supplied prediction results are audited.','Group disparities do not prove discrimination.']}

def simulate(details,model,threshold):
    if model not in details:raise ValueError('Unknown model.')
    d=details[model];p=(d['scores']>=threshold).astype(int)
    return {'model':model,'threshold':threshold,'metrics':evaluate(d['actual'],p,d['scores']),
            'fairness':group_audit(d['actual'],p,d['groups'],[c for c in PROTECTED if c in d['groups']]),
            'note':'Threshold scenario on existing holdout probabilities; not a future forecast.'}
