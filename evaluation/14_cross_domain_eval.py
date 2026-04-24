#!/usr/bin/env python3
"""
PriceBench 跨领域验证 — Task C (估价) 快速验证
对所有跨领域数据运行 0-shot 价格估算，验证 "LLM 不懂价格" 是否跨领域成立
"""
import json, os, sys, time, re, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx
from datetime import datetime

DIR = os.path.dirname(os.path.abspath(__file__))
CROSS = os.path.join(DIR, 'data', 'cross_domain')
OUT = os.path.join(DIR, 'results')

# ========== Model Configs ==========
OPENROUTER_KEY = os.environ.get('OPENROUTER_KEY', 'YOUR_OPEN_ROUTER_KEY')
OPENROUTER_URL = 'https://openrouter.ai/api/v1'

MODELS = {
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
    'gemini-2.5-flash': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'model': 'gemini-2.5-flash',
        'display': 'Gemini 2.5 Flash',
    },
    'gpt-4o-mini': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://api.openai.com/v1',
        'model': 'gpt-4o-mini',
        'display': 'GPT-4o-mini',
    },
    'mistral-large': {
        'api_key': OPENROUTER_KEY,
        'base_url': OPENROUTER_URL,
        'model': '@preset/mistral-large',
        'display': 'Mistral Large',
    },
    'llama-3.1-70b': {
        'api_key': OPENROUTER_KEY,
        'base_url': OPENROUTER_URL,
        'model': '@preset/llama-3-1-70-b',
        'display': 'Llama 3.1 70B',
    },
    'claude-sonnet': {
        'api_key': os.environ.get('API_KEY', 'YOUR_API_KEY'),
        'base_url': 'https://api.anthropic.com',
        'model': 'claude-sonnet-4-20250514',
        'display': 'Claude Sonnet 4',
    },
}


def get_client(model_key):
    cfg = MODELS[model_key]
    return OpenAI(
        api_key=cfg['api_key'],
        base_url=cfg['base_url'],
        timeout=httpx.Timeout(60.0, connect=15.0),
    )

def call_llm(client, model_name, prompt, max_retries=5):
    is_claude = 'claude' in model_name.lower()

    for attempt in range(max_retries):
        try:
            if is_claude:
                # Always use native Anthropic API for Claude models
                api_key = next(v['api_key'] for v in MODELS.values() if v['model'] == model_name)
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                payload = {
                    "model": model_name,
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}]
                }
                r = httpx.post("https://api.anthropic.com/v1/messages",
                               headers=headers, json=payload, timeout=60.0)
                if r.status_code == 529:
                    wait = min(20 * (attempt + 1), 60)
                    print(f"  [529 overloaded] waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                content = r.json()['content'][0]['text'].strip()
                if content:
                    return content
            else:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=100)
                content = (resp.choices[0].message.content or '').strip()
                if content:
                    return content
            time.sleep(1.0)
        except Exception as e:
            print(f"API Error ({model_name}): {e}")
            if attempt < max_retries - 1:
                time.sleep(min(2 ** (attempt + 1), 30))
            else:
                return ""
    return ""

def run_batch(client, model_name, prompts, concurrency=8, label=""):
    results = [None] * len(prompts)
    total = len(prompts)
    done_count = [0]
    start_time = time.time()
    lock = threading.Lock()

    def process(idx, prompt):
        resp = call_llm(client, model_name, prompt)
        with lock:
            done_count[0] += 1
            c = done_count[0]
            if c % 20 == 0 or c == total:
                elapsed = time.time() - start_time
                rate = c / elapsed if elapsed > 0 else 0
                eta = (total - c) / rate if rate > 0 else 0
                print(f"    {label} [{c}/{total}] ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
        return idx, resp

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(process, i, p): i for i, p in enumerate(prompts)}
        for f in as_completed(futs):
            idx, resp = f.result()
            results[idx] = resp

    return results


def make_prompt(item):
    """生成价格估算 prompt"""
    domain_desc = {
        'consumer_electronics': 'consumer electronics',
        'used_car': 'used car',
        'luxury_goods': 'pre-owned luxury goods',
        'home_appliance': 'home appliance',
    }
    domain = domain_desc.get(item['domain'], item['domain'])
    currency = item['currency']

    prompt = f"""You are a pricing expert. Estimate the fair market price for the following {domain} item.

Item: {item['item_name']}
Category: {item.get('category', 'N/A')}
Specifications: {item.get('specs', 'N/A')}"""

    if item['domain'] == 'used_car':
        prompt += f"\nYear: {item.get('year', 'N/A')}"
        prompt += f"\nMileage: {item.get('mileage', 'N/A')}"

    prompt += f"\n\nOutput ONLY a number (price in {currency}). No explanation, no currency symbol."
    return prompt


def parse_number(resp):
    if not resp:
        return 0
    resp = resp.replace(',', '').replace('$', '').replace('¥', '').replace('元', '').replace('万', '')
    m = re.findall(r'[-+]?\d*\.?\d+', resp)
    return float(m[0]) if m else 0


def calc_metrics(y_true, y_pred):
    yt = np.array(y_true, float)
    yp = np.array(y_pred, float)
    # 过滤掉 0 预测
    mask = yp > 0
    if mask.sum() == 0:
        return {'PRED25': 0, 'PRED50': 0, 'MdAPE': 999, 'valid_n': 0}
    yt, yp = yt[mask], yp[mask]
    mre = np.abs(yp - yt) / np.maximum(np.abs(yt), 0.01)
    return {
        'PRED25': round(float(np.mean(mre <= 0.25)), 4),
        'PRED50': round(float(np.mean(mre <= 0.50)), 4),
        'MdAPE': round(float(np.median(mre)), 4),
        'MAE': round(float(np.mean(np.abs(yp - yt))), 2),
        'valid_n': int(mask.sum()),
    }


def main():
    # 选择要跑的模型
    model_keys = list(MODELS.keys())
    if len(sys.argv) > 1 and sys.argv[1] != 'all':
        model_keys = [k for k in sys.argv[1:] if k in MODELS]

    # 加载所有跨领域数据 (新 4 领域)
    all_data = {}
    for fname in ['electronics_raw.json', 'used_cars_raw.json', 'luxury_goods_raw.json', 'appliances_raw.json']:
        path = os.path.join(CROSS, fname)
        if os.path.exists(path):
            with open(path) as f:
                items = json.load(f)
            domain = items[0]['domain'] if items else fname.split('_')[0]
            all_data[domain] = items

    print("PriceBench 跨领域验证 — Task C (估价)")
    print("=" * 60)
    for domain, items in all_data.items():
        print(f"  {domain}: {len(items)} 条")
    print(f"  总计: {sum(len(v) for v in all_data.values())} 条")
    print(f"  模型: {', '.join(model_keys)}")
    print("=" * 60, flush=True)

    results = {}

    for model_key in model_keys:
        cfg = MODELS[model_key]
        display = cfg['display']
        client = get_client(model_key)
        model_name = cfg['model']
        results[display] = {}

        for domain, items in all_data.items():
            label = f"{display} {domain}"
            print(f"\n▶ {label} | n={len(items)}", flush=True)

            prompts = [make_prompt(item) for item in items]
            t0 = time.time()
            preds_raw = run_batch(client, model_name, prompts, concurrency=8, label=label)
            elapsed = time.time() - t0

            y_true = [item['price'] for item in items]
            y_pred = [parse_number(r or '') for r in preds_raw]

            metrics = calc_metrics(y_true, y_pred)
            results[display][domain] = metrics
            print(f"  ✓ {label} 完成 | {elapsed:.1f}s | PRED25={metrics['PRED25']:.4f} | MdAPE={metrics['MdAPE']:.4f}", flush=True)

    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    save_path = os.path.join(OUT, f'cross_domain_validation_{ts}.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已保存: {save_path}")

    # 打印总表
    print("\n" + "=" * 70)
    print("Cross-Domain Validation: PRED(0.25) Summary")
    print("=" * 70)
    domains = list(all_data.keys())
    header = f"{'Model':<20}" + "".join(f"{d:<20}" for d in domains) + f"{'Mean':<10}"
    print(header)
    print("-" * len(header))

    for model, model_results in results.items():
        vals = []
        row = f"{model:<20}"
        for d in domains:
            v = model_results.get(d, {}).get('PRED25', 0)
            vals.append(v)
            row += f"{v:<20.4f}"
        row += f"{np.mean(vals):<10.4f}"
        print(row)

    # ====== Popularity Gradient Analysis ======
    print("\n" + "=" * 70)
    print("Popularity Gradient: Popular vs Niche PRED(0.25)")
    print("=" * 70)
    
    gradient_results = {}
    for model_key in model_keys:
        cfg = MODELS[model_key]
        display = cfg['display']
        gradient_results[display] = {}
        
    # Re-compute by popularity for each domain
    for domain, items in all_data.items():
        pop_items = [i for i in items if i.get('popularity', 'popular') == 'popular']
        niche_items = [i for i in items if i.get('popularity') == 'niche']
        if not niche_items:
            continue
        print(f"\n  {domain}: popular={len(pop_items)}, niche={len(niche_items)}")
    
    print(f"\n{'Model':<20} {'Domain':<22} {'Popular P25':<15} {'Niche P25':<15} {'Gap':<10}")
    print("-" * 82)
    
    # Store per-item predictions for gradient analysis
    for model_key in model_keys:
        cfg = MODELS[model_key]
        display = cfg['display']
        client = get_client(model_key)
        model_name = cfg['model']
        
        for domain, items in all_data.items():
            pop_items = [i for i in items if i.get('popularity', 'popular') == 'popular']
            niche_items = [i for i in items if i.get('popularity') == 'niche']
            if not niche_items or not pop_items:
                continue
            
            # Use cached results from the first run
            pop_true = [i['price'] for i in pop_items]
            niche_true = [i['price'] for i in niche_items]
            
            pop_prompts = [make_prompt(i) for i in pop_items]
            niche_prompts = [make_prompt(i) for i in niche_items]
            
            pop_preds = run_batch(client, model_name, pop_prompts, concurrency=8, label=f"{display} {domain} pop")
            niche_preds = run_batch(client, model_name, niche_prompts, concurrency=8, label=f"{display} {domain} niche")
            
            pop_pred_vals = [parse_number(r or '') for r in pop_preds]
            niche_pred_vals = [parse_number(r or '') for r in niche_preds]
            
            pop_m = calc_metrics(pop_true, pop_pred_vals)
            niche_m = calc_metrics(niche_true, niche_pred_vals)
            
            gap = pop_m['PRED25'] - niche_m['PRED25']
            print(f"{display:<20} {domain:<22} {pop_m['PRED25']:<15.4f} {niche_m['PRED25']:<15.4f} {gap:<10.4f}")
            
            gradient_results[display][domain] = {
                'popular': pop_m, 'niche': niche_m, 'gap': round(gap, 4)
            }
    
    # Save gradient results
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    grad_path = os.path.join(OUT, f'popularity_gradient_{ts}.json')
    with open(grad_path, 'w', encoding='utf-8') as f:
        json.dump(gradient_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Gradient 分析已保存: {grad_path}")


if __name__ == '__main__':
    main()
