"""CREDVION: reproducible holdout evaluation of UCI credit default data."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
TARGET = 'default payment next month'

def load_data(path=BASE/'dataset.xls'):
    df = pd.read_excel(path, header=1, engine='xlrd')
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f'Missing expected outcome column: {TARGET}. Columns: {list(df.columns)}')
    original = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    df = df.drop_duplicates(subset=['ID']) if 'ID' in df else df.drop_duplicates()
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df = df[df[TARGET].isin([0, 1])].copy()
    return df, {'original_rows':original,'rows_after_cleaning':len(df),'removed_rows':original-len(df),'missing_cells':int(df.isna().sum().sum())}

def rates(y, p):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return {'count':int(len(y)),'actual_defaults':int(sum(y)),'accuracy':round(float(accuracy_score(y,p)),4),
            'false_positive_rate':round(float(fp/(fp+tn)),4) if fp+tn else None,
            'false_negative_rate':round(float(fn/(fn+tp)),4) if fn+tp else None,
            'false_positives':int(fp),'false_negatives':int(fn),'true_positives':int(tp),'true_negatives':int(tn)}

def run_audit():
    df,cleaning=load_data()
    # Sensitive columns are excluded from training but used to audit errors.
    protected=['SEX','AGE','EDUCATION','MARRIAGE']
    drop=[TARGET,'ID']+[c for c in protected if c in df]
    X=df.drop(columns=drop,errors='ignore')
    y=df[TARGET].astype(int)
    train,test=train_test_split(np.arange(len(df)),test_size=.25,random_state=42,stratify=y)
    numeric=X.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in X if c not in numeric]
    steps=[]
    if numeric: steps.append(('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),numeric))
    if categorical: steps.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),categorical))
    model=Pipeline([('preprocess',ColumnTransformer(steps)),('classifier',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42))])
    model.fit(X.iloc[train],y.iloc[train]);pred=model.predict(X.iloc[test]);scores=model.predict_proba(X.iloc[test])[:,1]
    overall=rates(y.iloc[test].to_numpy(),pred)
    overall.update({'precision':round(float(precision_score(y.iloc[test],pred,zero_division=0)),4),
                    'recall':round(float(recall_score(y.iloc[test],pred,zero_division=0)),4),
                    'f1':round(float(f1_score(y.iloc[test],pred,zero_division=0)),4),
                    'roc_auc':round(float(roc_auc_score(y.iloc[test],scores)),4)})
    held=df.iloc[test].reset_index(drop=True)
    audits={}
    groups={
        'Sex':held['SEX'].map({1:'Male (coded 1)',2:'Female (coded 2)'}) if 'SEX' in held else None,
        'Age band':pd.cut(held['AGE'],bins=[0,25,35,45,55,150],labels=['≤25','26–35','36–45','46–55','56+']) if 'AGE' in held else None,
        'Education code':held['EDUCATION'].astype(str) if 'EDUCATION' in held else None,
        'Marriage code':held['MARRIAGE'].astype(str) if 'MARRIAGE' in held else None,
    }
    yt=y.iloc[test].to_numpy()
    for name,group in groups.items():
        if group is None:continue
        group=group.reset_index(drop=True)
        records=[]
        for label in sorted(group.dropna().unique(),key=str):
            mask=(group==label).to_numpy()
            r=rates(yt[mask],pred[mask]);r['group']=str(label)
            r['predicted_high_risk_rate']=round(float(pred[mask].mean()),4)
            records.append(r)
        audits[name]=records
    return {'brand':'CREDVION','dataset':'UCI Default of Credit Card Clients', 'cleaning':cleaning,
        'dataset_rows':len(df),'train_rows':len(train),'test_rows':len(test),
        'target_definition':'1 = default next month; 0 = no default',
        'method':'Stratified 75/25 holdout, seed 42; median/mode imputation fitted on train only; logistic regression with balanced class weights',
        'protected_columns_excluded_from_training':protected,
        'overall':overall,'audits':audits,
        'caveats':['Observed group differences do not establish discrimination or causation.',
          'This is a historical Taiwan credit-card dataset; not a production lending decision system.',
          'Groups with few defaults may have unstable error rates.',
          'Excluding protected attributes does not eliminate proxy effects.',
          'Use human review and legal/contextual assessment before policy decisions.']}

if __name__=='__main__':
    import json
    print(json.dumps(run_audit(),indent=2))
