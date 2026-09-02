import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from verantyx.cross_store import CrossStore
from verantyx.consensus import (ConsensusConfig, SearchState, evaluate, _enumerate_moves, _apply_move,
    visible_axes, axis_energy, global_hypotheses, run_consensus, copy_shell, N_SECTIONS, query_content)
from verantyx.cross import AXES, FACE_SLOTS
from verantyx.consensus_store import build_shell_from_store, candidates_for_query, _MassView, _ranked_facets
from verantyx.face_roles import CORE_FACE

SENTS = [
 ("defense has imminence",5),("defense has unlawfulness",5),("defense has proportionality",4),
 ("defense has necessity",3),("defense has statute",3),("defense has impunity",2),("defense has right",1),
 ("emergency has danger",5),("emergency has statute",4),("emergency has subsidiarity",3),("emergency has life",3),("emergency has impunity",2),
 ("negligence has duty",5),("negligence has foreseeability",4),("negligence has damages",3),("negligence has civil",3),("negligence has statute",1),
 ("assault has statute",4),("assault has injury",4),("assault has unlawfulness",3),("assault has imprisonment",2),
 ("homicide has statute",4),("homicide has death",4),("homicide has intent",3),("homicide has imprisonment",2),
 ("prescription has invocation",4),("prescription has completion",3),("prescription has civil",3),("prescription has interruption",2),
]
def store():
    st=CrossStore()
    for s,n in SENTS:
        for _ in range(n): st.ingest_sentence(s)
    return st
def dump(sh): return {a:{f:sh.faces[a].get(f) for f in FACE_SLOTS} for a in AXES}|{"center":sh.center}

def trace(st, query, seed=None):
    qset,_=query_content(query)
    cores=candidates_for_query(st, query)
    out={"query":query,"qset":sorted(qset),"retrieved":cores,
         "placement":{c:{"all":_ranked_facets(st,c),"counts":dict(st.crosses[c])} for c in cores},"steps":[],"final":{}}
    if not cores:
        out["final"]={"verdict":"UNKNOWN_NO_EVIDENCE","text":""}; return out
    shell=build_shell_from_store(st,cores); out["shell0"]=dump(shell)
    cfg=ConsensusConfig(); masses=_MassView(st); _seed=seed or {}
    state=SearchState(shell=copy_shell(shell),rotation=int(_seed.get("rotation",0))%N_SECTIONS,
                      widened=bool(_seed.get("widened",False)),locks=set(_seed.get("locks") or ()))
    esc=False; moves=0
    def snap(state,cur):
        return {"rotation":state.rotation,"widened":state.widened,"locks":sorted(state.locks),
                "energies":{a:round(axis_energy(state.shell,a,qset,cfg,masses),3) for a in AXES if state.shell.faces[a].get(CORE_FACE)},
                "sections":[{"axis":c[0],"core":c[1],"energy":round(c[2],3)} if c else None for c in cur.candidates],
                "visible":[visible_axes(state,i,cfg) for i in range(N_SECTIONS)],
                "key":[round(float(k),3) for k in cur.key()],"agree_all":cur.agree_all,"shell":dump(state.shell)}
    while True:
        cur=evaluate(state,qset,cfg,masses); best=None; bk=cur.key(); bs=None
        for mv in _enumerate_moves(state):
            s2=_apply_move(state,mv); k2=evaluate(s2,qset,cfg,masses).key()
            if k2<bk: bk=k2; best=mv; bs=s2
        out["steps"].append(snap(state,cur))
        if best is not None:
            state=bs; moves+=1
            out["steps"][-1]["chosen"]={"move":best[0],"arg":list(best[1]) if isinstance(best[1],tuple) else best[1]}; continue
        if cur.n_active==0: v="UNKNOWN_NO_EVIDENCE"; break
        if cur.agree_all:
            if cur.deficit>0: v="UNKNOWN_INSUFFICIENT_EVIDENCE"; break
            h=global_hypotheses(state.shell,qset,cfg,masses)
            if len(h)>=2 and (h[0][2]-h[1][2])<=cfg.tie_delta: v="AMBIGUOUS"; break
            v="ANSWER"; break
        if cfg.allow_escape and not esc:
            esc=True; state=SearchState(shell=state.shell,rotation=state.rotation,widened=True,locks=set())
            out["steps"][-1]["chosen"]={"move":"escape","arg":None}; continue
        v="UNKNOWN_SECTION_DISAGREEMENT" if esc else "UNKNOWN_LOCAL_MINIMUM"; break
    ref=run_consensus(shell,query,cfg=cfg,masses=masses,seed_state=seed)
    assert ref.verdict==v,(ref.verdict,v)
    out["final"]={"verdict":v,"core":ref.core,"moves":moves,"escape":esc,"tokens":ref.tokens,
                  "text":ref.text,"carry":ref.as_dict()["carry_state"],"center":ref.state.shell.center if ref.state else None,
                  "hypotheses":ref.as_dict()["hypotheses"]}
    return out

st=store()
T={"ingest":[{"sentence":s,"core":s.split()[0],"facet":s.split()[2]} for s,_ in SENTS[:3]],
   "store":{c:dict(st.crosses[c]) for c in st.crosses},
   "answer":trace(st,"what has statute unlawfulness imminence"),
   "refuse":trace(st,"what is quantumflux"),
   "tie":trace(st,"what has imprisonment")}
T["revisit"]=trace(st,"what has statute unlawfulness imminence",seed=T["answer"]["final"]["carry"])
for k in ("answer","refuse","tie","revisit"):
    f=T[k]["final"]; print(k,f.get("verdict"),f.get("core"),"moves",f.get("moves"),"esc",f.get("escape"),"steps",len(T[k]["steps"]),T[k]["retrieved"],"|",f.get("text"))
json.dump(T,open(pathlib.Path(__file__).with_name('traces_en.json'),'w'),ensure_ascii=False)
