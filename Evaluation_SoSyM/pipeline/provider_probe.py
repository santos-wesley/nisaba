"""Low-cost probe: verify that candidate providers support what the pipeline needs
(chat + function-calling + structured output) before committing to a generator lineup."""
import os, sys
from pydantic import BaseModel
import nisaba_port as N

PROVIDERS = {
    "xai":      {"gen": "x-ai/grok-4.5",              "gp": {"reasoning_effort": "none"}},
    "google":   {"gen": "google/gemini-3.5-flash",   "gp": {"reasoning_effort": "none"}},
    "deepseek": {"gen": "deepseek/deepseek-v4-flash", "gp": {"reasoning_effort": "none"}},
    "zai":      {"gen": "z-ai/glm-5.2",               "gp": {"reasoning_effort": "none"}},
}
TOOL = [{"type": "function", "function": {"name": "add", "description": "add two ints",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]}}}]
class Point(BaseModel):
    x: int; y: int

def try_(label, fn):
    try:
        return label, fn()
    except Exception as e:
        return label, "ERR: " + str(e)[:120]

for fam, c in PROVIDERS.items():
    print(f"\n### {fam}  gen={c['gen']}")
    # 1 generator chat (reasoning none)
    def g_chat():
        r = N.chat([{"role": "user", "content": "Reply with exactly: OK"}], c["gen"], params=c["gp"],
                   max_tokens=50, step="probe.gen.chat")
        return repr(r.choices[0].message.content)
    print("  gen.chat      ->", try_("", g_chat)[1])
    # 2 generator tools
    def g_tools():
        r = N.chat([{"role": "user", "content": "Use the add tool to add 3 and 4."}], c["gen"], params=c["gp"],
                   tools=TOOL, tool_choice="auto", max_tokens=200, step="probe.gen.tools")
        tc = r.choices[0].message.tool_calls
        return [(t.function.name, t.function.arguments) for t in tc] if tc else "NO TOOL CALL"
    print("  gen.tools     ->", try_("", g_tools)[1])
    # 3 generator structured (instructor)
    def g_struct():
        p = N.structured([{"role": "user", "content": "Return the point x=1, y=2."}], c["gen"], Point,
                         params=c["gp"], max_tokens=800, step="probe.gen.struct")
        return f"x={p.x} y={p.y}"
    print("  gen.struct    ->", try_("", g_struct)[1])
print("\nPROBE DONE")
