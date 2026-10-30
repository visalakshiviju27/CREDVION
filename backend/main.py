"""CREDVION full-stack API. Local hackathon demo, single active workspace."""
import io, json, csv
from pathlib import Path
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
from platform_engine import read_table,train_three,audit_external,simulate,TARGET,PROTECTED

app=FastAPI(title='CREDVION Multi-Model Fairness Audit',version='2.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['*'],allow_headers=['*'])
STATE={'dataset':None,'filename':None,'training':None,'details':None,'external':None}
BASE=Path(__file__).parent

def fail(exc):
    if isinstance(exc,HTTPException):raise exc
    raise HTTPException(status_code=400,detail=str(exc))

async def get_upload(file):
    if not file.filename or not file.filename.lower().endswith(('.csv','.xls','.xlsx')):
        raise ValueError('Select a CSV, XLS or XLSX file.')
    blob=await file.read(20_000_001)
    if len(blob)>20_000_000:raise ValueError('File exceeds 20 MB limit.')
    return read_table(blob,file.filename)

def info(df):
    return {'rows':len(df),'columns':list(df.columns),'preview':json.loads(df.head(12).to_json(orient='records')),
            'missing':{str(k):int(v) for k,v in df.isna().sum().items()},'duplicates':int(df.duplicated().sum()),
            'types':{str(k):str(v) for k,v in df.dtypes.items()}}

@app.get('/api/health')
def health():return {'status':'ok','app':'CREDVION','version':'2.0'}

@app.get('/api/workspace')
def workspace():
    return {'filename':STATE['filename'],'dataset':info(STATE['dataset']) if STATE['dataset'] is not None else None,
            'training':STATE['training'],'external':STATE['external']}

@app.post('/api/dataset/default')
def default_dataset():
    try:
        path=BASE/'dataset.xls'
        if not path.exists():raise ValueError('Bundled dataset.xls is missing.')
        df=read_table(path.read_bytes(),path.name)
        STATE.update(dataset=df,filename='UCI Default of Credit Card Clients',training=None,details=None)
        return {'filename':STATE['filename'],'dataset':info(df)}
    except Exception as e:fail(e)

@app.post('/api/dataset/upload')
async def upload_dataset(file:UploadFile=File(...)):
    try:
        df=await get_upload(file)
        STATE.update(dataset=df,filename=file.filename,training=None,details=None)
        return {'filename':file.filename,'dataset':info(df)}
    except Exception as e:fail(e)

@app.get('/api/dataset/cleaned.csv')
def cleaned_csv():
    if STATE['dataset'] is None:raise HTTPException(400,'Load a dataset first.')
    df=STATE['dataset'].drop_duplicates()
    return Response(df.to_csv(index=False),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="credvion_deduplicated_dataset.csv"'})

@app.post('/api/models/train')
def train(payload:dict):
    try:
        if STATE['dataset'] is None:raise ValueError('Load a dataset first.')
        target=payload.get('target') or TARGET
        groups=payload.get('demographics',[])
        if not isinstance(groups,list):raise ValueError('Demographics must be an array.')
        report,details=train_three(STATE['dataset'],target,groups)
        STATE.update(training=report,details=details)
        return report
    except Exception as e:fail(e)

@app.post('/api/external/audit')
async def external(file:UploadFile=File(...),actual:str=Form(...),prediction:str=Form(...),positive_label:str=Form('1'),demographics:str=Form('[]'),probability:str=Form('')):
    try:
        df=await get_upload(file);groups=json.loads(demographics)
        report=audit_external(df,actual,prediction,groups,positive_label,probability or None)
        STATE['external']=report
        return report
    except Exception as e:fail(e)

@app.post('/api/external/preview')
async def external_preview(file:UploadFile=File(...)):
    try:return info(await get_upload(file))
    except Exception as e:fail(e)

@app.post('/api/simulate')
def what_if(payload:dict):
    try:
        if STATE['details'] is None:raise ValueError('Train models first.')
        threshold=float(payload.get('threshold',.5))
        if not 0<=threshold<=1:raise ValueError('Threshold must be between 0 and 1.')
        return simulate(STATE['details'],payload.get('model'),threshold)
    except Exception as e:fail(e)

@app.get('/api/report/{kind}.{format}')
def report(kind:str,format:str):
    if kind not in ['training','external']:raise HTTPException(400,'Unknown report type.')
    data=STATE[kind]
    if data is None:raise HTTPException(400,'No completed audit to export.')
    if format=='json':return Response(json.dumps(data,indent=2,ensure_ascii=False),media_type='application/json',headers={'Content-Disposition':f'attachment; filename="CREDVION_{kind}.json"'})
    if format=='csv':
        rows=[]
        if kind=='training':
            for name,model in data['models'].items():rows.append({'model':name,**model['metrics']})
        else:rows=[{'model':'External AI',**data['metrics']}]
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return Response(output.getvalue(),media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="CREDVION_{kind}.csv"'})
    if format=='pdf':
        try:
            from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            buf=io.BytesIO();doc=SimpleDocTemplate(buf);styles=getSampleStyleSheet();story=[Paragraph('CREDVION | AI Credit-Risk Audit',styles['Title']),Spacer(1,15)]
            story.append(Paragraph('Academic demonstration. Observed group disparities do not establish discrimination.',styles['Normal']))
            story.append(Spacer(1,12))
            models=data['models'] if kind=='training' else {'External AI':{'metrics':data['metrics'],'fairness':data['fairness']}}
            for name,entry in models.items():
                story.append(Paragraph(str(name),styles['Heading2']))
                metrics=entry['metrics'];table=Table([['Metric','Value']]+[[str(k),str(v)] for k,v in metrics.items()],colWidths=[230,220]);table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d6edbc')),('GRID',(0,0),(-1,-1),.3,colors.grey)]));story.append(table);story.append(Spacer(1,12))
                for g,a in entry['fairness'].items():
                    story.append(Paragraph(f'{g}: FPR gap {a["fpr_gap"]}; FNR gap {a["fnr_gap"]}; group counts: '+', '.join(f'{r["group"]}={r["count"]}' for r in a['rows']),styles['Normal']))
                story.append(Spacer(1,12))
            story.append(Paragraph(str(data.get('method','')),styles['Normal']))
            for caveat in data.get('caveats',[]):story.append(Paragraph('• '+caveat,styles['Normal']))
            doc.build(story)
            return Response(buf.getvalue(),media_type='application/pdf',headers={'Content-Disposition':f'attachment; filename="CREDVION_{kind}.pdf"'})
        except Exception as e:fail(e)
    raise HTTPException(400,'Choose pdf, csv or json.')
