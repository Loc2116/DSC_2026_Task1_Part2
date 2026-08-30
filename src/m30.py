import json,sys
sys.path.append("/mnt/user-data/uploads/project_DSC")
from rerank_from_d import blend_bm25_first
U="/mnt/user-data/uploads/project_DSC"
dev=json.load(open(f"{U}/dev_300_locked.json",encoding="utf-8"))
S=json.load(open(f"{U}/Ketqua_E/scores_dev300_fusion_M20_K20.json"))
Q=[q for q in dev if q in S]; G={q:{str(a) for a in dev[q]["answer"]} for q in Q}
OD={q:sorted(S[q],key=lambda d:-S[q][d]["bm25"]) for q in Q}
PIPE={q:{d:max(v["ce"],v.get("ce_deep",v["ce"])) for d,v in S[q].items()} for q in Q}
def t5(q):
    rk=sorted(PIPE[q],key=lambda d:-PIPE[q][d])
    return blend_bm25_first(rk,OD[q],k=5,n_bm25=1)
sai=[q for q in Q if not (G[q]&set(t5(q)))]
print(f"{len(sai)} cau pipeline dang SAI\n")
print("thu hang cua gold theo `ce` (quyet dinh van ban nao duoc doc sau):")
import collections
c=collections.Counter()
for q in sai:
    ce=sorted(S[q],key=lambda d:-S[q][d]["ce"])
    r=[i for i,d in enumerate(ce,1) if d in G[q]]
    best=min(r) if r else 99
    c[ "1-10" if best<=10 else "11-20" if best<=20 else "21-30" if best<=30 else "31-50" if best<=50 else "ngoai ro"]+=1
for k in ("1-10","11-20","21-30","31-50","ngoai ro"): print(f"  {k:>9}: {c[k]}")
print(f"\n=> M=20 -> M=30 chi cham toi {c['21-30']} cau trong so {len(sai)} cau dang sai")
