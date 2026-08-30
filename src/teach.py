import json,sys
sys.path.append("/mnt/user-data/uploads/project_DSC")
from rerank_from_d import blend_bm25_first
U="/mnt/user-data/uploads/project_DSC"
dev=json.load(open(f"{U}/dev_300_locked.json",encoding="utf-8"))
S=json.load(open(f"{U}/Ketqua_E/scores_dev300_fusion_M20_K20.json"))
T=json.load(open(f"{U}/Ketqua_E/scores_hard15_qwen3rr8b.json"))
O=json.load(open(f"{U}/Ketqua_E/scores_hard15_ce_ours.json"))
Q=[q for q in dev if q in S]; G={q:{str(a) for a in dev[q]["answer"]} for q in Q}
OD={q:sorted(S[q],key=lambda d:-S[q][d]["bm25"]) for q in Q}
P={q:{d:max(v["ce"],v.get("ce_deep",v["ce"])) for d,v in S[q].items()} for q in Q}
t5=lambda q:blend_bm25_first(sorted(P[q],key=lambda d:-P[q][d]),OD[q],k=5,n_bm25=1)[:5]
sai=[q for q in Q if not (G[q]&set(t5(q)))]
print(f"18 cau dang sai; hard15 co {len(T)} cau")
gia=set(T)&set(sai); print(f"GIAO NHAU: {len(gia)} cau\n")
def top(sc,q,k=5): return sorted(sc[q],key=lambda d:-sc[q][d])[:k]
if gia:
    print("tren cac cau giao nhau — thay 8B co keo gold vao top-5 khong?")
    n8=n0=0
    for q in sorted(gia):
        a=bool(G[q]&set(top(T,q))); b=bool(G[q]&set(top(O,q)))
        n8+=a; n0+=b
        print(f"  {q:>8}  ce_ours:{'DUNG' if b else 'sai '}  8B:{'DUNG' if a else 'sai '}")
    print(f"\n  ce_ours {n0}/{len(gia)} · 8B {n8}/{len(gia)}")
print(f"\n{len(set(sai)-set(T))} cau dang sai CHUA duoc thay cham -> can chay bo sung")
print("  qid:", sorted(set(sai)-set(T)))
