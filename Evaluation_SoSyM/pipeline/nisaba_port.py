"""
nisaba_port.py -- local, provider-agnostic port of the Nisaba pipeline,
with all definitions loaded verbatim from Nisaba_(SoSym_Journal).ipynb.

Design:
  * All PROMPTS, the reconstruction SCHEMA (MPDeclareModel), the function-calling
    TOOLS, and the METRIC functions are loaded VERBATIM from the notebook by exec-ing
    the pure-definition cells -- nothing is transcribed by hand.
  * The Colab / CrewAI / google.colab plumbing is replaced by a thin OpenRouter layer
    (openai SDK -> https://openrouter.ai/api/v1) plus faithful single-call replicas of
    the CrewAI narrative agent.
"""
import os, re, json, math, time, hashlib, logging, random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment

from pydantic import (BaseModel, Field, conlist, conint, ValidationError,
                      field_validator, model_validator)
from typing import List, Optional, Dict, Union, Literal
from enum import Enum

from Declare4Py.ProcessModels.DeclareModel import DeclareModel
from openai import OpenAI
import instructor

logger = logging.getLogger("nisaba_port")            # referenced by schema validators (cell 20)
logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
logger.setLevel(logging.WARNING)
for _noisy in ("openai", "httpx", "httpcore", "instructor", "LiteLLM", "urllib3",
               "sentence_transformers", "transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Nisaba_(SoSym_Journal).ipynb")

# ---------------------------------------------------------------------------
# Derived-statistic helpers (inter-rater agreement, paired effect size, directional consistency),
# shared by run_test.py and understandability_test.py.
# ---------------------------------------------------------------------------
def gwet_ac1(a, b, categories=None):
    """Gwet's AC1 inter-rater agreement (2 raters), robust to the answer-prevalence imbalance that makes
    Cohen's kappa paradoxical here (high raw agreement but kappa near 0)."""
    n = min(len(a), len(b))
    if n == 0:
        return None
    a, b = list(a)[:n], list(b)[:n]
    cats = categories if categories is not None else sorted(set(a) | set(b))
    q = len(cats)
    if q < 2:
        return 1.0
    pa = sum(1 for x, y in zip(a, b) if x == y) / n
    pi = {k: (sum(1 for x in a if x == k) + sum(1 for x in b if x == k)) / (2 * n) for k in cats}
    pe = sum(pi[k] * (1 - pi[k]) for k in cats) / (q - 1)
    return 1.0 if pe >= 1 else round((pa - pe) / (1 - pe), 3)


def raw_agreement(a, b):
    n = min(len(a), len(b))
    return None if n == 0 else round(sum(1 for x, y in zip(list(a)[:n], list(b)[:n]) if x == y) / n, 3)


def cohens_d_paired(x, y):
    """Standardized paired effect size d_z = mean(x - y) / sd(x - y)."""
    d = [xi - yi for xi, yi in zip(x, y)]
    n = len(d)
    if n < 2:
        return None
    m = sum(d) / n
    sd = math.sqrt(sum((di - m) ** 2 for di in d) / (n - 1))
    return None if sd == 0 else round(m / sd, 2)


def direction_consistency(x, y, cmp="lt"):
    """(count, n) of pairs where x < y (cmp='lt') or x > y ('gt')."""
    n = min(len(x), len(y))
    c = sum(1 for xi, yi in zip(list(x)[:n], list(y)[:n]) if (xi < yi if cmp == "lt" else xi > yi))
    return (c, n)

# ---------------------------------------------------------------------------
# 1. Load verbatim definitions from the notebook
# ---------------------------------------------------------------------------
def _notebook_cells(path=NOTEBOOK):
    nb = json.load(open(path, encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in nb["cells"]]

# The pure-definition cells are located by CONTENT signature (not absolute index), so this
# survives cell insertions/removals in the notebook (e.g. adding or removing cells).
# Cells are exec'd in ASCENDING index order, so a later cell's redefinition wins the name
# (decltotextassistant: cell 17 then 30; response_template/toolsdescription: 17 then 43/33).
_DEF_SIGS = [
    ("assign", "response_template"),          # 17 (v1) + 43 (final)
    ("assign", "decltotextassistant"),        # 17 (v1) + 30 (final)
    ("assign", "toolsdescription"),           # 17 (v1) + 33 (final)
    ("assign", "guidelines"),                 # 43 (RST constraint->relation map)
    ("assign", "decl_reconstruction_assistant"),  # 39
    ("class",  "MPDeclareModel"),             # 20 (structured-output schema)
    ("def",    "calculate_size_metric"),      # 78 (complexity metrics block)
    ("def",    "compute_semantic_distance"),  # 81 (multi-perspective semantic distance)
]
def _cell_defines(src, kind, name):
    if kind == "assign":
        return re.search(r"(?m)^\s*" + re.escape(name) + r"\s*=", src) is not None
    return (f"{kind} {name}") in src

def load_defs():
    """Exec the notebook's pure-definition cells (found by content) in a controlled namespace."""
    cells = _notebook_cells()
    ns = dict(BaseModel=BaseModel, Field=Field, conlist=conlist, conint=conint,
              ValidationError=ValidationError, field_validator=field_validator,
              model_validator=model_validator, List=List, Optional=Optional, Dict=Dict,
              Union=Union, Literal=Literal, Enum=Enum, logger=logger, np=np, pd=pd, nx=nx,
              math=math, Counter=Counter, re=re, os=os, json=json,
              linear_sum_assignment=linear_sum_assignment, DeclareModel=DeclareModel)
    wanted = sorted({i for i, src in enumerate(cells)
                     for kind, name in _DEF_SIGS if _cell_defines(src, kind, name)})
    for idx in wanted:
        exec(compile(cells[idx], f"<notebook cell {idx}>", "exec"), ns)
    return ns

DEFS = load_defs()
decltotextassistant       = DEFS["decltotextassistant"]
decl_reconstruction_assistant = DEFS["decl_reconstruction_assistant"]
guidelines                = DEFS["guidelines"]
response_template         = DEFS["response_template"]
toolsdescription          = DEFS["toolsdescription"]
MPDeclareModel            = DEFS["MPDeclareModel"]
calculate_metrics         = DEFS["calculate_metrics"]
save_metrics_to_csv       = DEFS["save_metrics_to_csv"]
compute_semantic_distance = DEFS["compute_semantic_distance"]
# per-model complexity primitives (used to walk only the model .json files, not the run-level jsons)
calculate_size_metric                  = DEFS["calculate_size_metric"]
calculate_density_metric               = DEFS["calculate_density_metric"]
calculate_separability_metric          = DEFS["calculate_separability_metric"]
calculate_constraint_variability_metric = DEFS["calculate_constraint_variability_metric"]

# ---------------------------------------------------------------------------
# 2. Declare4Py helpers (verbatim from notebook cells 11, 86, 38)
# ---------------------------------------------------------------------------
class ExtendedDeclareModel(DeclareModel):
    def get_decl_attributes(self):
        attributes = []
        for attr_name, attr_obj in self.parsed_model.attributes_list.items():
            attr_value = ""
            if attr_obj.attr_value:
                attr_value = ": " + attr_obj.attr_value.value_original
            attributes.append(f"{attr_name}{attr_value}")
        return attributes

    def get_decl_binds(self):
        binds = []
        for event_type, events in self.parsed_model.events.items():
            for event_name, event_obj in events.items():
                if event_obj.attributes:
                    bound_attrs = ", ".join(a.get_name() for a in event_obj.attributes.values())
                    binds.append(f"bind {event_name}: {bound_attrs}")
        return binds

def format_prompt(declare_model_lines):
    model = "\n".join(declare_model_lines)
    return ("New model to generate description of activities, attributes, binds "
            "and constraints based in the template and examples: \n" + model)

# NOTE: mirror of the notebook's process_intermediate_descriptions (cell ~32); keep byte-identical.
# Not loaded via DEFS: that notebook cell is not pure -- it also runs
# `... = process_intermediate_descriptions(intermediary_tool_calls)`, which references an undefined
# name, so exec-ing the cell inside load_defs() would raise NameError and break import.
def process_intermediate_descriptions(tool_calls):
    """Tuple version (notebook cell 32): returns (model_string, structured_dict)."""
    model = {"Activities": [], "Attributes": [], "Binds": [], "Constraints": []}
    for item in tool_calls:
        func_args = json.loads(item.function.arguments)
        for key, value in func_args.items():
            bucket = model.get(key.capitalize())
            if bucket is None or not isinstance(value, list):
                continue
            for v in value:
                if isinstance(v, dict):
                    bucket.append({"name": v.get("name", ""), "description": v.get("description", "")})
                elif isinstance(v, str):
                    bucket.append({"name": v, "description": ""})
    model_string = ""
    for key, items in model.items():
        model_string += f"## {key}\n"
        for item in items:
            model_string += f"- **{item['name']}**: {item['description']}\n"
        model_string += "\n"
    return model_string, model

# ---------------------------------------------------------------------------
# 3. OpenRouter LLM layer (replaces the Colab/litellm/CrewAI plumbing)
# ---------------------------------------------------------------------------
BASE_URL = "https://openrouter.ai/api/v1"
LLM_TIMEOUT = float(os.environ.get("NISABA_LLM_TIMEOUT", "240"))   # per-call hard timeout (s)
_client = None
def client():
    global _client
    if _client is None:
        import httpx
        # explicit timeout + max_retries=0: a hung provider fails fast; our own _retry handles it.
        # raise the httpx connection pool so many worker threads do not contend for keepalive slots.
        _client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL,
                         timeout=LLM_TIMEOUT, max_retries=0,
                         http_client=httpx.Client(limits=httpx.Limits(max_connections=256,
                                                                      max_keepalive_connections=256)))
    return _client

import threading
_LOG_PATH = None
_LOG_LOCK = threading.Lock()
_TL = threading.local()                     # per-thread step context (thread-safe under parallel providers)
def set_log_path(p): global _LOG_PATH; _LOG_PATH = p
def set_step(**kw):                          # MERGE into this thread's context
    _TL.ctx = {**getattr(_TL, "ctx", {}), **kw}
def clear_step(): _TL.ctx = {}
def _ctx(): return getattr(_TL, "ctx", {})

def _reasoning_extra(params):
    extra = {}
    eff = (params or {}).get("reasoning_effort")
    if eff == "none":   extra["reasoning"] = {"enabled": False}
    elif eff:           extra["reasoning"] = {"effort": eff}
    prov = (params or {}).get("provider")     # OpenRouter provider routing (e.g. round-robin across a model's endpoints)
    if prov:            extra["provider"] = prov
    return extra

def _log(model, messages, response, usage, extra):
    if not _LOG_PATH:
        return
    rec = {"t": time.time(), **_ctx(), "model": model, "params_extra": extra,
           "messages": messages, "response": response, "usage": usage}
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with _LOG_LOCK:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)

def _retry(fn, n=3, base=3):
    last = None
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            last = e
            # A max_tokens truncation is deterministic (the model just needs more room, which the fixed
            # budget will not give) -> retrying only burns another full generation. Fail fast instead.
            # A timeout is likewise non-retryable here: a hung/too-slow call will just time out again,
            # so retrying 3x only stalls up to ~12 min on a single call. Fail fast on it too.
            if ("max_tokens" in str(e).lower() or type(e).__name__ == "IncompleteOutputException"
                    or "timeout" in type(e).__name__.lower() or "timed out" in str(e).lower()):
                raise
            print(f"    [retry {i+1}/{n}] {type(e).__name__}: {str(e)[:180]}")
            time.sleep(base * (i + 1))
    raise last

def _usage(r):
    u = getattr(r, "usage", None)
    if not u:
        return None
    return u.model_dump() if hasattr(u, "model_dump") else dict(u)

def chat(messages, model, params=None, tools=None, tool_choice=None,
         max_tokens=16000, step=""):
    """Single chat completion via OpenRouter. Reasoning models omit temperature;
    non-reasoning gen calls use temperature=0."""
    extra = _reasoning_extra(params)
    eff = (params or {}).get("reasoning_effort")
    def call():
        kw = dict(model=model, messages=messages, max_tokens=max_tokens)
        if not eff or eff == "none":
            kw["temperature"] = 0
            kw["top_p"] = 1.0
            kw["seed"] = 42
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice or "auto"
        if extra:
            kw["extra_body"] = extra
        r = client().chat.completions.create(**kw)
        m = r.choices[0].message
        # when plain text is expected, an empty content == failure (reasoning ate the token
        # budget, or a transient provider hiccup) -> raise so _retry re-issues the call.
        if tools is None and not getattr(m, "content", None):
            raise RuntimeError(f"empty content from {model} (step={step})")
        return r
    r = _retry(call)
    msg = r.choices[0].message
    tcs = [{"name": t.function.name, "arguments": t.function.arguments}
           for t in (msg.tool_calls or []) if getattr(t, "function", None)] if getattr(msg, "tool_calls", None) else None
    _log(model, messages, {"content": msg.content, "tool_calls": tcs}, _usage(r),
         {"step": step, **extra})
    return r

def structured(messages, model, response_model, params=None, max_tokens=16000, step=""):
    """Structured output via instructor (robust across the rich MPDeclareModel schema)."""
    extra = _reasoning_extra(params)
    ic = instructor.from_openai(client())
    def call():
        kw = dict(model=model, response_model=response_model, messages=messages,
                  max_tokens=max_tokens, max_retries=1)
        if extra:
            kw["extra_body"] = extra
        return ic.chat.completions.create(**kw)
    parsed = _retry(call)
    _log(model, messages, {"parsed": parsed.model_dump()}, None, {"step": step, **extra})
    return parsed

# ---------------------------------------------------------------------------
# 4. Pipeline steps
# ---------------------------------------------------------------------------
def _sanitize_decl(text):
    """Drop declare4py-fragile lines so parse_from_string does not raise on imperfect Terpsichora models:
    (1) empty binds ('bind X:'); (2) orphan attribute-domain lines whose attribute is never bound by any
    bind ('Attr: value' -> declare4py raises 'Unable to find attribute'). Same rule the metric layer uses."""
    lines = text.split("\n")
    bound = set()
    for line in lines:
        s = line.strip()
        if s.startswith("bind ") and ":" in s:
            for a in s.split(":", 1)[1].split(","):
                a = a.strip()
                if a:
                    bound.add(a)
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("bind "):
            after = s.split(":", 1)[1].strip() if ":" in s else ""
            if after == "":
                continue
            out.append(line)
            continue
        if (":" in s and "[" not in s and not s.startswith("activity ")
                and DeclareModel.is_events_attrs_value_definition(s)):
            lhs = [a.strip() for a in s.split(":", 1)[0].split(",")]
            if any(a not in bound for a in lhs):
                continue
            if "between" in s.lower():                      # (3) range domain w/o two numeric
                nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)", s.split(":", 1)[1])
                if len(nums) < 2:                            # declare4py parse_attr_value -> IndexError
                    continue
        out.append(line)
    return "\n".join(out)


def _robust_parse(decl_string):
    """Parse a (sanitized) model; if declare4py rejects a bind whose event it cannot match (a fragility on
    some hyphenated activity names), drop that bind line and retry, so no imperfect Terpsichora model
    crashes the pipeline. Returns the parsed ExtendedDeclareModel."""
    decl = decl_string
    dm = ExtendedDeclareModel()
    for _ in range(30):
        decl = _sanitize_decl(decl)          # re-clean each pass (dropping a bind can orphan its domain)
        dm = ExtendedDeclareModel()
        try:
            dm.parse_from_string(decl)
            return dm
        except ValueError as e:
            m = re.search(r"Unable to find the event or activity (.+)$", str(e))
            if not m:
                raise
            bad = m.group(1).strip()
            kept = [l for l in decl.split("\n")
                    if not (l.strip().startswith("bind ")
                            and l.strip()[len("bind"):].split(":", 1)[0].strip() == bad)]
            if len(kept) == len(decl.split("\n")):
                raise
            decl = "\n".join(kept)
        except KeyError as e:                    # dedupe a duplicated 'activity' declaration
            if "Multiple times the same event name" not in str(e):
                raise
            seen = set()
            deduped = []
            for l in decl.split("\n"):
                if l.strip().startswith("activity "):
                    a = l.strip()[len("activity "):].strip()
                    if a in seen:
                        continue
                    seen.add(a)
                deduped.append(l)
            if len(deduped) == len(decl.split("\n")):
                raise
            decl = "\n".join(deduped)
    return dm


def gen_intermediary(decl_string, gen_model, gen_params):
    """MP-Declare -> intermediary description via function calling (faithful path).
    Falls back to structured output if the model does not return the 4 tool calls."""
    dm = _robust_parse(decl_string)
    messages = [{"role": "system", "content": decltotextassistant},
                {"role": "user", "content": format_prompt(dm.declare_model_lines)}]
    r = chat(messages, gen_model, params=gen_params, tools=toolsdescription,
             tool_choice="auto", step="generation.intermediary")
    tool_calls = [t for t in (r.choices[0].message.tool_calls or []) if getattr(t, "function", None)]
    if not tool_calls:
        # fallback: single structured object with all four construct lists
        class _Item(BaseModel):
            name: str; description: str
        class _Interm(BaseModel):
            activities: List[_Item] = Field(default_factory=list)
            attributes: List[_Item] = Field(default_factory=list)
            binds: List[_Item] = Field(default_factory=list)
            constraints: List[_Item] = Field(default_factory=list)
        parsed = structured(messages, gen_model, _Interm, params=gen_params,
                            step="generation.intermediary.fallback")
        tool_calls = [SimpleNamespace(function=SimpleNamespace(arguments=parsed.model_dump_json()))]
    return process_intermediate_descriptions(tool_calls)   # (string, dict)

def reconstruct(intermediary_string, gen_model, gen_params):
    messages = [{"role": "system", "content": decl_reconstruction_assistant},
                {"role": "user", "content": "Create a MP-Declare model of the following process "
                 f"using the knowledge from the assistant: \n {intermediary_string}"}]
    return structured(messages, gen_model, MPDeclareModel, params=gen_params, step="reconstruction")

def _compose_agent_system(role, goal, backstory):
    """Faithful-in-content replica of the system prompt CrewAI composes from an Agent."""
    return (f"You are {role}. {backstory}\nYour personal goal is: {goal}\n"
            "You must return your complete answer directly, not a summary.")

def narrative_legacy(intermediary_string, gen_model, gen_params):
    """CrewAI 'Narrative Generator' agent + CreateNarrative task (cells 44-45), as one call."""
    system = _compose_agent_system(
        role="Narrative Generator",
        goal="Generate an interleaved narrative of given process model's constraints.",
        backstory=("You are an expert in narrative generation, specializing in creating cohesive "
                   "descriptions of MP-Declare process models's constraints using rhetorical relations "
                   "from the Rethorical Structure Theory."))
    task = ("The MP-Declare model you will use to generate the narrative: \n" + intermediary_string +
            "\nUse the following rhetorical relations for the given group of MP-Declare constraints: \n"
            + guidelines)
    expected = ("The interleaved narrative of the given MP-Declare process model's constraints. "
                "Strictly follow the following response template: \n" + response_template)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task + "\n\nExpected output:\n" + expected}]
    r = chat(messages, gen_model, params=gen_params, step="generation.narrative.legacy")
    return r.choices[0].message.content

def zero_shot(decl_string, gen_model, gen_params):
    # Notebook feeds the declare4py-canonical lines (model = "\n".join(declare_model.declare_model_lines),
    # cell ~1539) to the zero-shot prompt, NOT the raw text. Mirror that; fall back to sanitized raw
    # text only if the model cannot be parsed (crash-safe on imperfect Terpsichora models).
    try:
        model_text = "\n".join(_robust_parse(decl_string).declare_model_lines)
    except Exception:
        model_text = _sanitize_decl(decl_string)
    messages = [{"role": "user", "content":
                 "Generate a natural language description of the following Multi-Perspective "
                 "Declare model:\n" + model_text}]
    r = chat(messages, gen_model, params=gen_params, step="zeroshot")
    return r.choices[0].message.content

# ---------------------------------------------------------------------------
# 5. Embedder for semantic distance (gte-large per paper; robust fallback)
# ---------------------------------------------------------------------------
def load_embedder():
    """Load gte-large-en-v1.5 (paper's embedder). Applies the position_ids buffer fix if
    the transformers RoPE meta-load left it corrupted. Falls back to MiniLM if gte fails."""
    from sentence_transformers import SentenceTransformer
    def _fix_position_ids(model):
        try:
            import torch
            for mod in model.modules():
                buf = getattr(mod, "position_ids", None)
                if buf is not None and hasattr(buf, "shape"):
                    fixed = torch.arange(buf.shape[-1]).reshape(buf.shape)
                    mod.register_buffer("position_ids", fixed, persistent=False)
        except Exception as e:
            print("    [embedder] position_ids fix skipped:", e)
    try:
        m = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        try:
            m.encode(["warmup sentence"])
        except Exception:
            _fix_position_ids(m)
            m.encode(["warmup sentence"])
        return m, "Alibaba-NLP/gte-large-en-v1.5", 0.2   # tau calibrated for gte-large
    except Exception as e:
        print("    [embedder] gte-large failed, falling back to MiniLM:", str(e)[:200])
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return m, "sentence-transformers/all-MiniLM-L6-v2", 0.5
