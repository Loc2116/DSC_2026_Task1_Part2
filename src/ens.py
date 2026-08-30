import json,os,sys,glob,itertools
sys.path.append("/mnt/user-data/uploads/project_DSC")
from rerank_from_d import blend_bm25_first
U="/mnt/user-data/uploads/project_DSC"
dev=json.load(open(f"{U}/dev_300_locked.json",encoding="utf-8"))
ORDER=json.load(open(f"{U}/Ketqua_E/order300_fusion.json",encoding="utf-8"))
qids=[q for q in dev if q in ORDER]
GOLD={q:{str(a) for a in dev[q]["answer"]} for q in qids}
D="/home/claude/work/r12/outputs"
S={os.path.basename(p)[16:-5]:json.load(open(p)) for p in glob.glob(f"{D}/scores_ablation_*.json")}
S.pop("thanhtan")                      # trùng bit với AITeamVN
def ev(per,n=0):
    r5=r1=0.0
    for q in qids:
        rk=sorted(per[q],key=lambda d:-per[q][d])
        p=blend_bm25_first(rk,ORDER[q],k=10,n_bm25=n)
        r5+=len(GOLD[q]&set(p[:5]))/len(GOLD[q]); r1+=len(GOLD[q]&set(p[:1]))/len(GOLD[q])
    return r5/len(qids), r1/len(qids)
def rrf(tags,k=60):
    out={}
    for q in qids:
        acc={}
        for t in tags:
            rk=sorted(S[t][q],key=lambda d:-S[t][q][d])
            for i,d in enumerate(rk,1): acc[d]=acc.get(d,0)+1.0/(k+i)
        out[q]=acc
    return out
base5,base1=ev(S["AITeamVN"]); print(f"AITeamVN mot minh: R@5={base5:.4f} R@1={base1:.4f}\n")
tags=sorted(S)
# oracle: cau nao co IT NHAT 1 model dat gold vao top5
hit={t:set() for t in tags}
for t in tags:
    for q in qids:
        rk=sorted(S[t][q],key=lambda d:-S[t][q][d])[:5]
        if GOLD[q]&set(rk): hit[t].add(q)
uni=set().union(*hit.values())
print(f"ORACLE 8 model (>=1 model dung top-5): {len(uni)/len(qids):.4f}  ·  AITeamVN {len(hit['AITeamVN'])/len(qids):.4f}")
print(f"  cau AITeamVN SAI ma model khac DUNG: {len(uni-hit['AITeamVN'])}")
print(f"  cau KHONG model nao dung        : {len(set(qids)-uni)}\n")
res=[]
for r in (2,3,4):
    for c in itertools.combinations(tags,r):
        if "AITeamVN" not in c: continue
        a,b=ev(rrf(c)); res.append((a,b,c))
res.sort(reverse=True)
print("TOP 12 to hop RRF (co AITeamVN):")
for a,b,c in res[:12]: print(f"  R@5={a:.4f} ({(a-base5)*100:+.2f})  R@1={b:.4f}  {'+'.join(c)}")
