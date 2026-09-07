import csv,sys,io,os,re
HDR=None
rows=[]
meta=[]
for path in sys.argv[1:]:
    for line in open(path, errors='replace'):
        if line.startswith('build_commit,'):
            HDR=next(csv.reader([line])); continue
        if line.startswith('"') and HDR:
            r=next(csv.reader([line]))
            if len(r)==len(HDR): rows.append(dict(zip(HDR,r)))
        elif line.startswith(('=====','exit:','disk_read','GUARD','REFUSE','args:','SKIPPED','--- stderr')):
            meta.append(line.rstrip())
print(f"{'model':28} {'ncmoe':>5} {'thr':>4} {'ctk':>5} {'fa':>3} {'test':>14} {'tok/s':>9} {'sd':>7}")
for r in rows:
    if r['n_prompt']!='0': test=f"pp{r['n_prompt']}"
    else: test=f"tg{r['n_gen']}"
    if r['n_depth']!='0': test+=f"@d{r['n_depth']}"
    print(f"{os.path.basename(r['model_filename'])[:28]:28} {r['n_cpu_moe']:>5} {r['n_threads']:>4} {r['type_k']:>5} {r['flash_attn']:>3} {test:>14} {float(r['avg_ts']):>9.2f} {float(r['stddev_ts']):>7.2f}")
print()
for m in meta: print(m)
