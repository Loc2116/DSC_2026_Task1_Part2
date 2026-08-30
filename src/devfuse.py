import json,sys
sys.path.append("/mnt/user-data/uploads/project_DSC")
from rerank_from_d import blend_bm25_first
U="/mnt/user-data/uploads/project_DSC"
dev=json.load(open(f"{U}/dev_300_locked.json",encoding="utf-8"))
SF=json.load(open(f"{U}/Ketqua_E/scores_dev300_fusion_M20_K20.json"))   # ro FUSION
SD=json.load(open(f"{U}/Ketqua_E/scores_dev300_deep_M20_K20.json"))     # ro BM25 (he 17/08)
Q=[q for q in dev if q in SF and q in SD]
G={q:{str(a) for a in dev[q]["answer"]} for q in Q}
def rank(S,q,n,k=10):
    per={d:max(v["ce"],v.get("ce_deep",v["ce"])) for d,v in S[q].items()}
    od=sorted(S[q],key=lambda d:-S[q][d]["bm25"])
    return blend_bm25_first(sorted(per,key=lambda d:-per[d]),od,k=k,n_bm25=n)
r5=lambda f:sum(len(G[q]&set(f(q)[:5]))/len(G[q]) for q in Q)/len(Q)
print(f"{len(Q)} cau chung")
for n in (1,2):
    print(f"  ro FUSION n={n}: {r5(lambda q:rank(SF,q,n)):.4f}   ro BM25 n={n}: {r5(lambda q:rank(SD,q,n)):.4f}")
def fus(wa,wb,k=60):
    def f(q):
        acc={}
        for w,r in ((wa,rank(SF,q,1)),(wb,rank(SD,q,2))):
            for i,d in enumerate(r,1): acc[d]=acc.get(d,0)+w/(k+i)
        return sorted(acc,key=lambda d:-acc[d])
    return f
best=r5(lambda q:rank(SF,q,1))
print(f"\nTRON hai he (RRF)  [he tot nhat mot minh = {best:.4f}]")
for w in ((1,1),(2,1),(3,1)):
    v=r5(fus(*w)); print(f"  w={w}: {v:.4f}  ({(v-best)*100:+.2f} diem)")
d=sum(set(rank(SF,q,1)[:5])!=set(rank(SD,q,2)[:5]) for q in Q)
t1=sum(rank(SF,q,1)[0]!=rank(SD,q,2)[0] for q in Q)
print(f"\nhai he lech nhau: TAP5 {d}/{len(Q)} cau · hang-1 {t1} cau  (giong hinh public: 756/1000, 218)")
