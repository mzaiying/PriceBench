#!/usr/bin/env python3
"""
PriceBench LLM 评测脚本 v3 - 线程池并发版
使用 ThreadPoolExecutor + openai SDK 实现并发，兼容所有模型
用法:
  python3 05_llm_eval_async.py                     # 跑全部模型
  python3 05_llm_eval_async.py gemini-1.5-flash gpt-4o-mini  # 只跑指定模型
  python3 05_llm_eval_async.py --concurrency 15    # 调整并发数
  python3 05_llm_eval_async.py --resume results/llm_eval_v3_xxx.json  # 断点续跑
"""
import os, json, time, re, sys
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(DIR, 'data', 'bench_v2.json')
OUT = os.path.join(DIR, 'results')
os.makedirs(OUT, exist_ok=True)

# ========== API Keys ==========
OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY', 'YOUR_OPEN_ROUTER_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1'

# ========== Model Config ==========
MODELS = {
    'mistral-large': {
        'api_key': OPENROUTER_KEY, 'base_url': OPENROUTER_URL,
        'model': '@preset/mistral-large',
        'display': 'Mistral Large',
    },
    'llama-3.1-70b': {
        'api_key': OPENROUTER_KEY, 'base_url': OPENROUTER_URL,
        'model': '@preset/llama-3-1-70-b',
        'display': 'Llama 3.1 70B',
    },
    'gemini-2.5-flash': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'model': 'gemini-2.5-flash',
        'display': 'Gemini 2.5 Flash',
    },
    'claude-sonnet-official': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://api.anthropic.com/v1/',
        'model': 'claude-sonnet-4-20250514',
        'display': 'Claude Sonnet 4',
    },
    'deepseek-v3': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://api.deepseek.com/v1',
        'model': 'deepseek-chat',
        'display': 'DeepSeek-V3',
    },
    'qwen2.5-72b': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        'model': 'qwen-plus',
        'display': 'Qwen2.5-72B',
    },
    'gpt-4o-mini': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
        'display': 'GPT-4o-mini',
    },
}

DEFAULT_CONCURRENCY = 10

# ========== Thread-safe LLM Call using openai SDK ==========
_clients = {}
_client_lock = threading.Lock()

def get_client(model_key):
    """线程安全地获取/复用 OpenAI client"""
    with _client_lock:
        if model_key not in _clients:
            from openai import OpenAI
            import httpx
            cfg = MODELS[model_key]
            _clients[model_key] = OpenAI(
                api_key=cfg['api_key'],
                base_url=cfg['base_url'],
                timeout=httpx.Timeout(60.0, connect=15.0),
            )
        return _clients[model_key]

def call_llm_single(model_key, prompt, max_retries=3):
    """单次 LLM 调用（线程安全，用 openai SDK）"""
    cfg = MODELS[model_key]
    client = get_client(model_key)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            content = (resp.choices[0].message.content or '').strip()
            if content:
                return content
            time.sleep(1)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1), 15)
                time.sleep(wait)
            else:
                return ""
    return ""

def run_batch_concurrent(model_key, prompts, concurrency=10, label=""):
    """使用线程池并发运行一批 prompt，返回有序结果"""
    results = [None] * len(prompts)
    total = len(prompts)
    done_count = [0]  # mutable for closure
    start_time = time.time()
    lock = threading.Lock()

    def process(idx, prompt):
        resp = call_llm_single(model_key, prompt)
        with lock:
            done_count[0] += 1
            c = done_count[0]
            if c % 50 == 0 or c == total:
                elapsed = time.time() - start_time
                rate = c / elapsed if elapsed > 0 else 0
                eta = (total - c) / rate if rate > 0 else 0
                print(f"    {label} [{c}/{total}] ({rate:.1f}/s, ETA {eta:.0f}s) last={resp[:30]}...")
        return idx, resp

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(process, i, p): i for i, p in enumerate(prompts)}
        for future in as_completed(futures):
            idx, resp = future.result()
            results[idx] = resp

    return results

def load_data():
    with open(BENCH, 'r') as f:
        return json.load(f)

# ========== Prompt 模板 ==========
def task_a_prompt(d, shots=None):
    base = """You are an IT procurement cost reviewer. Given the following IT item information, predict whether the expert reviewer would:
- "reduced": reduce the price (cost cut)
- "increased": increase the price
- "unchanged": keep the price unchanged

Respond with ONLY one word: reduced, increased, or unchanged."""
    if shots:
        base += "\n\nExamples:\n"
        for s in shots:
            base += f"\nItem: {s['item_name']}, Category: {s['category']}, Price: {s['total_original']}万元"
            if s.get('spec_original','').strip(): base += f", Spec: {s['spec_original'][:80]}"
            base += f"\nAnswer: {s['direction']}\n"
    item = f"Item: {d['item_name']}, Category: {d['category']}, Price: {d['total_original']}万元"
    if d.get('spec_original','').strip(): item += f", Spec: {d['spec_original'][:100]}"
    return base + f"\n\n{item}\nAnswer:"

def task_b_prompt(d, shots=None):
    base = """You are an IT procurement cost reviewer. Given the following IT item, estimate the adjustment rate (percentage of price reduction by the expert reviewer).
Output ONLY a number (the percentage), e.g., "15.5" means 15.5% reduction. Output "0" if no reduction expected."""
    if shots:
        base += "\n\nExamples:\n"
        for s in shots:
            base += f"\nItem: {s['item_name']}, Category: {s['category']}, Price: {s['total_original']}万元"
            if s.get('spec_original','').strip(): base += f", Spec: {s['spec_original'][:80]}"
            base += f"\nRate: {s.get('adjustment_rate',0)}\n"
    item = f"Item: {d['item_name']}, Category: {d['category']}, Price: {d['total_original']}万元"
    if d.get('spec_original','').strip(): item += f", Spec: {d['spec_original'][:100]}"
    return base + f"\n\n{item}\nRate:"

def task_c_prompt(d, shots=None):
    base = """You are an IT procurement expert. Estimate the fair market price (in 万元/10k CNY) for the following IT item based on its name and specifications.
Output ONLY a number (the price in 万元), e.g., "25.5"."""
    if shots:
        base += "\n\nExamples:\n"
        for s in shots:
            base += f"\nItem: {s['item_name']}, Category: {s['category']}"
            if s.get('spec_original','').strip(): base += f", Spec: {s['spec_original'][:80]}"
            base += f"\nFair Price: {s.get('total_reduced', s.get('total_original',0))}\n"
    item = f"Item: {d['item_name']}, Category: {d['category']}"
    if d.get('spec_original','').strip(): item += f", Spec: {d['spec_original'][:100]}"
    return base + f"\n\n{item}\nFair Price:"

def task_d_prompt(d, shots=None):
    cats = "设备购置费, 其他, 数据工程费, 应用软件开发费, 运行维护费, 配套工程费, 链路租用费"
    base = f"""Classify the following IT procurement item into one of these cost categories:
{cats}
Output ONLY the category name in Chinese, nothing else."""
    if shots:
        base += "\n\nExamples:\n"
        for s in shots:
            base += f"\nItem: {s['item_name']}, Price: {s['total_original']}万元"
            if s.get('spec_original','').strip(): base += f", Spec: {s['spec_original'][:60]}"
            base += f"\nCategory: {s['category']}\n"
    item = f"Item: {d['item_name']}, Price: {d['total_original']}万元"
    if d.get('spec_original','').strip(): item += f", Spec: {d['spec_original'][:100]}"
    return base + f"\n\n{item}\nCategory:"

def task_e_prompt(d, shots=None):
    base = """Check if the following arithmetic is correct: unit_price × quantity = total_price.
If the arithmetic is CORRECT, respond "normal".
If the arithmetic is WRONG (the numbers don't multiply correctly), respond "anomaly".
Output ONLY one word: normal or anomaly."""
    if shots:
        base += "\n\nExamples:\n"
        for s in shots:
            up=float(s.get('unit_price_original',0)); qty=float(s.get('qty_original',0)); total=float(s.get('total_original',0))
            label='anomaly' if abs(up*qty-total)/max(total,0.01)>0.01 else 'normal'
            base += f"\nUnit price: {up}万, Quantity: {qty}, Total: {total}万\nAnswer: {label}\n"
    return base + f"\n\nUnit price: {d.get('unit_price_original',0)}万, Quantity: {d.get('qty_original',0)}, Total: {d.get('total_original',0)}万\nAnswer:"

def task_f_prompt(d, shots=None):
    base = """You are an IT procurement reviewer. Given the item details below, predict whether the expert reviewer would modify the specifications.
Respond with ONLY one word: "changed" if specs will be modified, or "unchanged" if specs will stay the same."""
    if shots:
        base += "\n\nExamples:\n"
        for s in shots:
            so=s.get('spec_original','').strip(); sr=s.get('spec_reduced','').strip()
            label='changed' if so and sr and so!=sr else 'unchanged'
            base += f"\nItem: {s['item_name']}, Category: {s['category']}, Price: {s['total_original']}万, Spec: {so[:60]}\nAnswer: {label}\n"
    return base + f"\n\nItem: {d['item_name']}, Category: {d['category']}, Price: {d['total_original']}万, Spec: {d['spec_original'][:100]}\nAnswer:"

# ========== Parsers ==========
def parse_direction(resp):
    resp=resp.lower().strip().strip('"').strip("'")
    for w in ['reduced','increased','unchanged']:
        if w in resp: return w
    return 'unchanged'

def parse_number(resp):
    m=re.search(r'[-+]?\d*\.?\d+', resp.strip())
    return float(m.group()) if m else 0.0

def parse_category(resp):
    for c in ['设备购置费','其他','数据工程费','应用软件开发费','运行维护费','配套工程费','链路租用费']:
        if c in resp.strip(): return c
    return '其他'

def parse_binary(resp, pos='anomaly'):
    resp=resp.lower().strip().strip('"').strip("'")
    if pos in resp: return pos
    if pos=='changed' and 'change' in resp: return 'changed'
    return 'normal' if pos=='anomaly' else 'unchanged'

# ========== Metrics ==========
def calc_clf_multi(y_true, y_pred, classes):
    n=len(y_true); acc=sum(t==p for t,p in zip(y_true,y_pred))/n if n else 0
    f1s,pc=[],{}
    for cls in classes:
        tp=sum(t==cls and p==cls for t,p in zip(y_true,y_pred))
        fp=sum(t!=cls and p==cls for t,p in zip(y_true,y_pred))
        fn=sum(t==cls and p!=cls for t,p in zip(y_true,y_pred))
        sup=sum(t==cls for t in y_true)
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
        f1=2*pr*re/(pr+re) if pr+re else 0
        f1s.append(f1); pc[cls]={'P':round(pr,4),'R':round(re,4),'F1':round(f1,4),'n':sup}
    return {'Accuracy':round(acc,4),'Macro-F1':round(np.mean(f1s),4),'per_class':pc}

def calc_clf_binary(y_true, y_pred, pos):
    n=len(y_true)
    tp=sum(t==pos and p==pos for t,p in zip(y_true,y_pred))
    fp=sum(t!=pos and p==pos for t,p in zip(y_true,y_pred))
    fn=sum(t==pos and p!=pos for t,p in zip(y_true,y_pred))
    tn=n-tp-fp-fn; acc=(tp+tn)/n if n else 0
    pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
    f1=2*pr*re/(pr+re) if pr+re else 0
    return {'Accuracy':round(acc,4),'Precision':round(pr,4),'Recall':round(re,4),'F1':round(f1,4)}

def calc_reg(pred, actual):
    pred,actual=np.array(pred,float),np.array(actual,float)
    ae=np.abs(pred-actual); mre=ae/np.maximum(np.abs(actual),0.01)
    return {'MAE':round(np.mean(ae),4),'RMSE':round(np.sqrt(np.mean(ae**2)),4),
            'PRED_10':round(np.mean(mre<=0.10),4),'PRED_25':round(np.mean(mre<=0.25),4)}

def select_shots(data, task, n=3, seed=42):
    np.random.seed(seed)
    if task=='A':
        return [([d for d in data if d['direction']==dir] or data)[np.random.randint(max(1,sum(1 for d in data if d['direction']==dir)))] for dir in ['reduced','increased','unchanged']]
    elif task=='E':
        valid=[d for d in data if d.get('unit_price_original') and d.get('qty_original') and d.get('total_original') and float(d['unit_price_original'])>0]
        normal=[d for d in valid if abs(float(d['unit_price_original'])*float(d['qty_original'])-float(d['total_original']))/max(float(d['total_original']),0.01)<=0.01]
        anomaly=[d for d in valid if abs(float(d['unit_price_original'])*float(d['qty_original'])-float(d['total_original']))/max(float(d['total_original']),0.01)>0.01]
        shots=[]
        if normal: shots.append(normal[np.random.randint(len(normal))])
        if anomaly: shots.append(anomaly[np.random.randint(len(anomaly))])
        if normal and len(shots)<3: shots.append(normal[np.random.randint(len(normal))])
        return shots
    elif task=='F':
        valid=[d for d in data if d.get('spec_original','').strip()]
        changed=[d for d in valid if d.get('spec_original','').strip()!=d.get('spec_reduced','').strip() and d.get('spec_reduced','').strip()]
        unchanged=[d for d in valid if not(d.get('spec_original','').strip()!=d.get('spec_reduced','').strip() and d.get('spec_reduced','').strip())]
        shots=[]
        if changed: shots.append(changed[np.random.randint(len(changed))])
        if unchanged: shots.append(unchanged[np.random.randint(len(unchanged))])
        if unchanged and len(shots)<3: shots.append(unchanged[np.random.randint(len(unchanged))])
        return shots
    else:
        pool=list(data); np.random.shuffle(pool); return pool[:n]

def run_task(model_key, task_id, data, mode='0-shot', concurrency=10):
    """并发评测单个 task"""
    cfg = MODELS[model_key]
    shots = select_shots(data, task_id) if mode=='3-shot' else None
    prompt_fn = {'A':task_a_prompt,'B':task_b_prompt,'C':task_c_prompt,
                 'D':task_d_prompt,'E':task_e_prompt,'F':task_f_prompt}[task_id]

    if task_id=='E':
        valid=[d for d in data if d.get('unit_price_original') and d.get('qty_original') and d.get('total_original') and float(d['unit_price_original'])>0 and float(d['qty_original'])>0 and float(d['total_original'])>0]
    elif task_id=='F':
        valid=[d for d in data if d.get('spec_original','').strip()]
    else:
        valid=data

    n = len(valid)
    display = cfg.get('display', model_key)
    label = f"{display} {mode} Task {task_id}"
    print(f"\n  ▶ {label} | n={n} | concurrency={concurrency}")

    # 生成所有 prompt
    prompts = [prompt_fn(d, shots) for d in valid]

    # 并发调用
    t0 = time.time()
    preds_raw = run_batch_concurrent(model_key, prompts, concurrency=concurrency, label=label)
    elapsed = time.time() - t0
    print(f"    ✓ {label} 完成 | {elapsed:.1f}s | {n/elapsed:.1f} 条/秒")

    # 计算指标
    if task_id=='A':
        y_true=[d['direction'] for d in valid]; y_pred=[parse_direction(r or '') for r in preds_raw]
        metrics=calc_clf_multi(y_true, y_pred, ['reduced','increased','unchanged'])
    elif task_id=='B':
        y_true=[d.get('adjustment_rate',0) or 0 for d in valid]; y_pred=[parse_number(r or '') for r in preds_raw]
        metrics=calc_reg(y_pred, y_true)
    elif task_id=='C':
        y_true=[float(d.get('total_reduced',d.get('total_original',0))) for d in valid]; y_pred=[parse_number(r or '') for r in preds_raw]
        metrics=calc_reg(y_pred, y_true)
    elif task_id=='D':
        y_true=[d.get('category','其他') for d in valid]; y_pred=[parse_category(r or '') for r in preds_raw]
        metrics=calc_clf_multi(y_true, y_pred, sorted(set(y_true)))
    elif task_id=='E':
        y_true=[]
        for d in valid:
            up=float(d['unit_price_original']); qty=float(d['qty_original']); total=float(d['total_original'])
            y_true.append('anomaly' if abs(up*qty-total)/max(total,0.01)>0.01 else 'normal')
        y_pred=[parse_binary(r or '','anomaly') for r in preds_raw]
        metrics=calc_clf_binary(y_true, y_pred, 'anomaly')
    elif task_id=='F':
        y_true=[]
        for d in valid:
            so=d.get('spec_original','').strip(); sr=d.get('spec_reduced','').strip()
            y_true.append('changed' if so and sr and so!=sr else 'unchanged')
        y_pred=[parse_binary(r or '','changed') for r in preds_raw]
        metrics=calc_clf_binary(y_true, y_pred, 'changed')

    print(f"    → {metrics}")
    return {'task':task_id,'model':model_key,'display':display,'mode':mode,
            'n':n,'metrics':metrics,'raw_sample':[(r or '')[:40] for r in preds_raw[:5]]}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', help='模型 key 列表')
    parser.add_argument('--concurrency', '-c', type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument('--resume', type=str, default=None, help='断点续跑的结果文件')
    parser.add_argument('--tasks', type=str, default='A,B,C,D,E,F', help='要跑的 task，逗号分隔')
    parser.add_argument('--modes', type=str, default='0-shot,3-shot', help='评测模式')
    args = parser.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    print(f"PriceBench LLM 评测 v3 (线程池并发) - {ts}")
    print(f"并发数: {args.concurrency}")
    print(f"{'='*60}")

    data = load_data()
    print(f"数据集: {len(data)} 条")

    models_to_run = args.models if args.models else list(MODELS.keys())
    tasks = args.tasks.split(',')
    modes = args.modes.split(',')

    # 断点续跑
    all_results = []
    done_keys = set()
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'r') as f:
            all_results = json.load(f)
        done_keys = {(r['model'], r['task'], r['mode']) for r in all_results if 'error' not in r}
        print(f"断点续跑: 已有 {len(all_results)} 组结果，跳过已完成")

    save_path = args.resume or os.path.join(OUT, f"llm_eval_v3_{ts}.json")

    total_jobs = sum(1 for mk in models_to_run if mk in MODELS for mode in modes for tid in tasks if (mk, tid, mode) not in done_keys)
    done_jobs = 0
    print(f"待跑: {total_jobs} 组 | 模型: {[MODELS[mk]['display'] for mk in models_to_run if mk in MODELS]}")
    print(f"{'='*60}")

    for mk in models_to_run:
        if mk not in MODELS:
            print(f"  跳过未知模型: {mk}"); continue
        display = MODELS[mk]['display']
        print(f"\n{'='*60}\n  模型: {display} ({mk})\n{'='*60}")

        for mode in modes:
            for tid in tasks:
                if (mk, tid, mode) in done_keys:
                    print(f"  ⏭ {display} {mode} Task {tid} (已完成，跳过)")
                    continue
                try:
                    r = run_task(mk, tid, data, mode, concurrency=args.concurrency)
                    all_results.append(r)
                    done_jobs += 1
                    with open(save_path, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
                    print(f"    💾 已保存 ({done_jobs}/{total_jobs})")
                except Exception as e:
                    print(f"    ❌ ERROR: {e}")
                    all_results.append({'task':tid,'model':mk,'display':display,'mode':mode,'error':str(e)})

    print(f"\n{'='*60}")
    print(f"  全部完成! {len(all_results)} 组结果 → {save_path}")
    print(f"{'='*60}")
    print(f"\n  {'Model':20s} {'Mode':8s} {'Task':6s} {'主指标':>12s}")
    print(f"  {'-'*55}")
    for r in all_results:
        if 'error' in r: continue
        m = r['metrics']
        if r['task'] in ['A','D']: val=f"MF1={m.get('Macro-F1',0):.4f}"
        elif r['task']=='B': val=f"MAE={m.get('MAE',0):.2f}"
        elif r['task']=='C': val=f"P25={m.get('PRED_25',0):.4f}"
        else: val=f"F1={m.get('F1',0):.4f}"
        print(f"  {r.get('display',r['model']):20s} {r['mode']:8s} {r['task']:6s} {val:>12s}")

if __name__ == '__main__':
    main()
