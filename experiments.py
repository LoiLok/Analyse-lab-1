"""Воспроизводимые эксперименты: python experiments.py (из любого каталога)."""
import csv
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from apriori import apriori, load_baskets, association_rules, sort_rules, format_rule

BASE = Path(__file__).resolve().parent
OUT = BASE / 'results'
CONFIDENCES = [.70, .75, .80, .85, .90, .95]
REPEATS = 15

def main():
    OUT.mkdir(exist_ok=True)
    baskets = load_baskets(BASE / 'baskets.csv')
    summary = {'baskets': len(baskets), 'items': len(set.union(*baskets)),
               'python': platform.python_version(), 'platform': platform.platform(),
               'processor': platform.processor(), 'repeats': REPEATS,
               'sha256': hashlib.sha256((BASE/'baskets.csv').read_bytes()).hexdigest(), 'runs': []}
    for support, tag in [(.10, 'support10'), (.001, 'support0_1')]:
        t = time.perf_counter()
        frequent = apriori(baskets, support)
        apriori_seconds = time.perf_counter() - t
        rows, raw = [], []
        for confidence in CONFIDENCES:
            association_rules(frequent, confidence)  # прогрев
            batch = 100 if support == .10 else 1
            samples = []
            for repeat in range(REPEATS):
                t = time.perf_counter_ns()
                for _ in range(batch):
                    rules = association_rules(frequent, confidence)
                elapsed = (time.perf_counter_ns()-t)/1e6/batch
                samples.append(elapsed)
                raw.append([confidence, repeat+1, batch, elapsed])
            rows.append({'confidence': confidence, 'rules': len(rules),
                         'median_ms': statistics.median(samples), 'min_ms': min(samples),
                         'max_ms': max(samples)})
            with (OUT/f'{tag}_rules_{round(confidence*100)}.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.writer(f); w.writerow(['antecedent','consequent','support','confidence','union_size','union_count'])
                for a,c,s,cf in sort_rules(rules,'support'):
                    w.writerow(['; '.join(a),'; '.join(c),s,cf,len(a)+len(c),round(s*len(baskets))])
            print(tag, rows[-1])
        with (OUT/f'{tag}_timings.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f); w.writerow(['confidence','repeat','batch','ms_per_call']); w.writerows(raw)
        for metric, ylabel, name in [('median_ms','Время поиска правил, мс','time'), ('rules','Количество правил','count')]:
            fig,ax=plt.subplots(figsize=(8,4))
            ys=[r[metric] for r in rows]
            ax.plot([c*100 for c in CONFIDENCES],ys,'o-',color='#205680')
            if metric=='median_ms':
                ax.fill_between([c*100 for c in CONFIDENCES],[r['min_ms'] for r in rows],[r['max_ms'] for r in rows],alpha=.15,label='Минимум и максимум 15 замеров'); ax.legend(fontsize=8)
            for x,y in zip(CONFIDENCES,ys): ax.annotate(f'{y:.3f}' if metric=='median_ms' else str(y),(x*100,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set(xlabel='Порог достоверности, %',ylabel=ylabel,title=f'Поддержка {support:.1%}')
            ax.margins(y=.25); ax.grid(alpha=.2); fig.tight_layout();fig.savefig(OUT/f'{tag}_{name}.png',dpi=180); plt.close(fig)
        summary['runs'].append({'support': support,'tag':tag,'frequent':len(frequent),'max_length':max(map(len,frequent),default=0),'apriori_seconds':apriori_seconds,'rows':rows})
        if support==.001:
            selected=sort_rules(association_rules(frequent,.95,7),'support')
            summary['selected_rules']=selected
            (OUT/'rules_max7.txt').write_text('\n'.join(format_rule(r) for r in selected),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()