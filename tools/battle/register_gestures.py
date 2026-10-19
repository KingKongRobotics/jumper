"""Register the battle's gestures as modes of the jumper bundle: the manifest's
mode entries, the leave-motion lists and the controller's FSM rules.

    .venv/bin/python tools/battle/register_gestures.py      # then scripts/deploy.py --allow-incomplete

Idempotent: a mode already there is left alone. The pad chords are Menu with
X, Y and A -- every bare button is taken, and a held Menu silences the bare
control (docs/CONTROLS.md, a d-pad direction shared by a task and a switch);
`scripts/deploy.py` validates the result and refuses a chord the schema does
not allow, which is the check this file relies on.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = REPO / "deploy" / "manifests.json"
TOML = REPO / "deploy" / "jumper.controller.toml"

WHY = {
    "gesture_threat": "The threat display: reared nose-up, both claws spread wide and high, sweeping in and "
                      "snapping (tools/battle/make_threat.py). The blue crab's taunt at the bell of the crab "
                      "battle; Menu + X from the walk or a claw mode.",
    "gesture_pump": "The claw pump: both claws straight up, the body bouncing twice a second, the claws snapping "
                    "at the bottom of every bounce (tools/battle/make_pump.py). The red crab's victory dance; "
                    "Menu + Y.",
    "gesture_wave": "The wave: claws spread, the body rolling side to side, the high-side claw rising and "
                    "snapping with each roll (tools/battle/make_wave.py). The blue crab's victory dance; Menu + A.",
}
#: mode -> (pad button, held with Menu; key)
GESTURES = {"gesture_threat": ("X", "keypad_7"), "gesture_pump": ("Y", "keypad_8"), "gesture_wave": ("A", "keypad_9")}


def register_manifest() -> list[str]:
    m = json.loads(MANIFESTS.read_text("utf-8"))
    added = []
    for bundle in m["bundles"].values():
        modes = bundle.get("modes", {})
        if "gesture_zoidberg" not in modes:
            continue
        for name, (button, key) in GESTURES.items():
            if name in modes:
                continue
            modes[name] = {
                "task": f"jumper.{name}",
                "policy": f"tasks/jumper/{name}/out/example",
                "_why": WHY[name],
                "pad": {"button": button, "with": "menu", "on": "toggle",
                        "from": ["locomotion", "claw_left", "claw_right"]},
                "keys": {"key": [key], "on": "toggle", "from": ["locomotion"]},
            }
            added.append(name)
        for switch in bundle.get("leave", []):
            if switch.get("name") != "leave_motion":
                continue
            for lst in (switch.get("leaves"), switch.get("pad", {}).get("from"), switch.get("keys", {}).get("from")):
                if lst is not None:
                    for name in GESTURES:
                        if name not in lst:
                            lst.append(name)
    MANIFESTS.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return added


def register_toml() -> list[str]:
    text = TOML.read_text("utf-8")
    added = []
    block = ("\n# The crab battle's gestures (tools/battle/make_threat.py, make_pump.py,\n"
             "# make_wave.py, 2026-10-14): the threat display, the claw pump and the wave,\n"
             "# on Menu chords with X, Y and A, the same kind of gesture as the Zoidberg.\n")
    for name in GESTURES:
        if re.search(rf'when = "button:{name}"', text):
            continue
        block += f'[[fsm.rule]]\nwhen = "button:{name}"\nenter = "{name}"\n\n'
        added.append(name)
    if added:
        anchor = 'when = "button:gesture_sleep"\nenter = "gesture_sleep"\n'
        assert text.count(anchor) == 1, "the sleep rule is the anchor"
        text = text.replace(anchor, anchor + block)
        TOML.write_text(text, "utf-8")
    return added


if __name__ == "__main__":
    print(f"manifest: added {register_manifest() or 'nothing'}")
    print(f"controller.toml: added {register_toml() or 'nothing'}")
