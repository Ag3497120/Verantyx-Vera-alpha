# 実エンジンの探索を一歩ずつ記録する(消費するのは run_consensus と同じ関数)
import json, sys
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from verantyx.cross_store import CrossStore
from verantyx.consensus import (ConsensusConfig, SearchState, evaluate, _enumerate_moves,
    _apply_move, visible_axes, axis_energy, global_hypotheses, run_consensus, copy_shell,
    N_SECTIONS, compose_answer)
from verantyx.cross import AXES, FACE_SLOTS
from verantyx.consensus_store import build_shell_from_store, ja_consensus_ask, _MassView, _ranked_facets, consensus_over_store
from verantyx.face_roles import facts_on_axis, CORE_FACE
from verantyx.lang import ja_content_runs

SENTS = [
 ("正当防衛 has 急迫不正", 5), ("正当防衛 has 侵害", 5), ("正当防衛 has 防衛行為", 4),
 ("正当防衛 has 相当性", 3), ("正当防衛 has 刑法", 3), ("正当防衛 has 不処罰", 2), ("正当防衛 has 権利", 1),
 ("緊急避難 has 危難", 5), ("緊急避難 has 刑法", 4), ("緊急避難 has 補充性", 3), ("緊急避難 has 生命", 3), ("緊急避難 has 不処罰", 2),
 ("過失 has 注意義務", 5), ("過失 has 予見可能性", 4), ("過失 has 損害賠償", 3), ("過失 has 民法", 3), ("過失 has 刑法", 1),
 ("傷害罪 has 刑法", 4), ("傷害罪 has 生理機能", 4), ("傷害罪 has 侵害", 3), ("傷害罪 has 拘禁刑", 2),
 ("殺人罪 has 刑法", 4), ("殺人罪 has 死", 4), ("殺人罪 has 故意", 3), ("殺人罪 has 拘禁刑", 2),
 ("時効 has 援用", 4), ("時効 has 完成", 3), ("時効 has 民法", 3), ("時効 has 中断", 2),
]

def store():
    st = CrossStore()
    for s, n in SENTS:
        for _ in range(n): st.ingest_sentence(s)
    return st

def shell_dump(sh):
    return {a: {f: sh.faces[a].get(f) for f in FACE_SLOTS} for a in AXES} | {"center": sh.center}

def trace(st, query, seed=None):
    r = ja_consensus_ask(st, query)
    cores = r.get("retrieved") or []
    out = {"query": query, "runs": ja_content_runs(query), "retrieved": cores,
           "placement": {c: {"all": _ranked_facets(st, c), "counts": dict(st.crosses[c])} for c in cores},
           "steps": [], "final": {}}
    if not cores:
        out["final"] = {"verdict": r["verdict"], "text": ""}
        return out
    shell = build_shell_from_store(st, cores)
    out["shell0"] = shell_dump(shell)
    cfg = ConsensusConfig(); masses = _MassView(st); qset = set(out["runs"])
    _seed = seed or {}
    state = SearchState(shell=copy_shell(shell), rotation=int(_seed.get("rotation",0))%N_SECTIONS,
                        widened=bool(_seed.get("widened",False)), locks=set(_seed.get("locks") or ()))
    escape_used=False; moves=0; verdict=None
    def snap(state, cur, label, move=None):
        return {"label": label, "move": move, "rotation": state.rotation, "widened": state.widened,
                "locks": sorted(state.locks),
                "energies": {a: round(axis_energy(state.shell,a,qset,cfg,masses),3) for a in AXES if state.shell.faces[a].get(CORE_FACE)},
                "sections": [ {"axis":c[0],"core":c[1],"energy":round(c[2],3)} if c else None for c in cur.candidates],
                "visible": [visible_axes(state,i,cfg) for i in range(N_SECTIONS)],
                "key": list(cur.key()), "agree_all": cur.agree_all, "shell": shell_dump(state.shell)}
    while True:
        cur = evaluate(state, qset, cfg, masses)
        best=None; bk=cur.key(); bs=None
        for mv in _enumerate_moves(state):
            s2=_apply_move(state,mv); k2=evaluate(s2,qset,cfg,masses).key()
            if k2<bk: bk=k2; best=mv; bs=s2
        out["steps"].append(snap(state, cur, "evaluate", None))
        if best is not None:
            state=bs; moves+=1
            out["steps"][-1]["chosen"]={"move":best[0],"arg":list(best[1]) if isinstance(best[1],tuple) else best[1]}
            continue
        if cur.n_active==0: verdict="UNKNOWN_NO_EVIDENCE"; break
        if cur.agree_all:
            if cur.deficit>0: verdict="UNKNOWN_INSUFFICIENT_EVIDENCE"; break
            hyps=global_hypotheses(state.shell,qset,cfg,masses)
            if len(hyps)>=2 and (hyps[0][2]-hyps[1][2])<=cfg.tie_delta: verdict="AMBIGUOUS"; break
            verdict="ANSWER"; break
        if cfg.allow_escape and not escape_used:
            escape_used=True
            state=SearchState(shell=state.shell,rotation=state.rotation,widened=True,locks=set())
            out["steps"][-1]["chosen"]={"move":"escape","arg":None}
            continue
        verdict="UNKNOWN_SECTION_DISAGREEMENT" if escape_used else "UNKNOWN_LOCAL_MINIMUM"; break
    ref = run_consensus(shell, query, cfg=cfg, masses=masses, qset_override=qset, seed_state=seed)
    assert ref.verdict==verdict, (ref.verdict, verdict)
    core = ref.core
    fin = {"verdict":verdict,"core":core,"moves":moves,"escape":escape_used,
           "tokens":ref.tokens, "carry": ref.as_dict()["carry_state"],
           "center": ref.state.shell.center if ref.state else None}
    if verdict=="ANSWER":
        facets=[t for t in ref.tokens if t!=core]
        fin["text"]= core + ("は"+"、".join(facets) if facets else "")
    out["final"]=fin
    return out

st=store()
ingest_demo=[]
for s,_ in SENTS[:3]:
    ingest_demo.append({"sentence":s, "runs":ja_content_runs(s)})
traces = {
 "ingest": ingest_demo,
 "store": {c: dict(st.crosses[c]) for c in st.crosses},
 "answer": trace(st, "刑法 侵害 防衛行為"),
 "refuse": trace(st, "超伝導とは"),
 "split":  trace(st, "拘禁刑"),
}
# 巡回: 同じ問いを終端配置つきで再訪
first=traces["answer"]["final"]["carry"]
traces["revisit"]=trace(st, "刑法 侵害 防衛行為", seed=first)
for k in ("answer","refuse","split","revisit"):
    t=traces[k]; print(k, t["final"].get("verdict"), t["final"].get("core"), "moves", t["final"].get("moves"), "esc", t["final"].get("escape"), "steps", len(t["steps"]), t["retrieved"])
    print("  ", t["final"].get("text"))
json.dump(traces, open(pathlib.Path(__file__).with_name('traces.json'),'w'), ensure_ascii=False)
