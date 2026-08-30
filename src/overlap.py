import json,os,sys,glob,itertools,random
sys.path.append("/mnt/user-data/uploads/project_DSC")
from rerank_from_d import blend_bm25_first
U="/mnt/user-data/uploads/project_DSC"
dev=json.load(open(f"{U}/dev_300_locked.json",encoding="utf-8"))
ORDER=json.load(open(f"{U}/Ketqua_E/order300_fusion.json",encoding="utf-8"))
qids=[q for q in dev if q in ORDER]
GOLD={q:{str(a) for a in dev[q]["answer"]} for q in qids}
FULL=json.load(open(f"{U}/Ketqua_E/scores_dev300_fusion_M20_K20.json"))
D="/home/claude/work/r12/outputs"
S={os.path.basename(p)[16:-5]:json.load(open(p)) for p in glob.glob(f"{D}/scores_ablation_*.json")}
S.pop("thanhtan")

def top5(per,q,n):
    rk=sorted(per[q],key=lambda d:-per[q][d])
    return blend_bm25_first(rk,ORDER[q],k=10,n_bm25=n)[:5]
def rrf1(tags,q,k=60):
    acc={}
    for t in tags:
        rk=sorted(S[t][q],key=lambda d:-S[t][q][d])
        for i,d in enumerate(rk,1): acc[d]=acc.get(d,0)+1.0/(k+i)
    return acc
# pipeline that ACTUALLY scored 0.9350: variant max = max(ce, ce_deep), n=1
PIPE={q:{d:max(v["ce"],v.get("ce_deep",v["ce"])) for d,v in FULL[q].items()} for q in qids}
ok=lambda s:{q for q in qids if GOLD[q]&set(s[q])}
Spipe={q:top5(PIPE,q,1) for q in qids}
Sai  ={q:top5(S["AITeamVN"],q,0) for q in qids}
E=("AITeamVN","bge-v2-m3","mMiniLMv2")
Sens ={q:top5({q:rrf1(E,q)},q,0) for q in qids}
P,A,N=ok(Spipe),ok(Sai),ok(Sens)
f=lambda s:len(s)/len(qids)
print(f"pipeline day du (max,n=1)   : {f(P):.4f}   [moc that = 0.9350]")
print(f"tang1 AITeamVN mot minh     : {f(A):.4f}")
print(f"tang1 ENSEMBLE 3 model      : {f(N):.4f}  ({(f(N)-f(A))*100:+.2f})\n")
moi=N-A
print(f"cau ensemble CUU them so voi AITeamVN tang1: {len(moi)}")
print(f"  trong do pipeline day du DA lam dung roi : {len(moi&P)}  <-- chong lap")
print(f"  pipeline VAN SAI (loi ich THAT su)       : {len(moi-P)}")
mat=A-N
print(f"cau ensemble LAM HONG (AITeamVN dung, ens sai): {len(mat)} · trong do pipeline dung: {len(mat&P)}")
print(f"\nNET tren nhung cau pipeline dang SAI: +{len(moi-P)} -{len(mat&P)} = {len(moi-P)-len(mat&P):+d} cau"
      f"  ({(len(moi-P)-len(mat&P))/len(qids)*100:+.2f} diem)")
print(f"\ntong cau pipeline sai: {len(qids)-len(P)}")

print("\n" + "="*70)
print("CONG M: ensemble co keo gold vao top-10 (suat doc sau) o 18 cau pipeline dang SAI khong?")
sai=[q for q in qids if q not in P]
ce={q:{d:v["ce"] for d,v in FULL[q].items()} for q in qids}
a=b=0
for q in sai:
    g10=set(sorted(ce[q],key=lambda d:-ce[q][d])[:10])
    e=rrf1(E,q); e10=set(sorted(e,key=lambda d:-e[d])[:10])
    ia,ib=bool(GOLD[q]&g10),bool(GOLD[q]&e10)
    a+=ia; b+=ib
    if ib and not ia: print(f"  {q}: ensemble KEO DUOC gold vao top-10 ma ce khong")
print(f"  ce hien tai : {a}/{len(sai)} cau co gold trong top-10")
print(f"  ensemble    : {b}/{len(sai)} cau")
