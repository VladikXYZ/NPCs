"""Interactive NPC/template debugging: python chat.py --reasoning mini."""
import argparse
import json
import os
import sys

from bench import ROOT, choose
from npc_runtime import generate, npc_messages
import utils


class InteractiveChat:
    def __init__(self, reasoning="off", npc_name=None, model_name=None, device_index=None):
        models = utils.get_models()
        if not models:
            raise ValueError("No visible NPC model weights installed; Supra is a hidden hardware probe.")
        if model_name:
            models = [m for m in models if model_name.casefold() in m["name"].casefold()]
            if len(models) != 1:
                raise ValueError("--model must match exactly one installed visible model.")
            self.model = models[0]
        else:
            self.model = choose(models, "Model", lambda m: m["name"])
        devices = utils.get_devices()
        if device_index is None:
            self.device = choose(devices, "Device", lambda d: f"{d['type']} | {d['name']}")
        elif 0 <= device_index < len(devices):
            self.device = devices[device_index]
        else:
            raise ValueError("Invalid device index.")
        npcs = json.loads((ROOT / "data_3npcs.json").read_text(encoding="utf-8"))
        if npc_name:
            selected = [npc for npc in npcs if npc["name"].casefold() == npc_name.casefold()]
            if not selected:
                raise ValueError(f"Unknown NPC: {npc_name}")
            self.npc = selected[0]
        else:
            self.npc = choose(npcs, "NPC", lambda n: f"{n['name']} ({n['profession']})")
        self.reasoning = reasoning
        os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"]) if self.device["type"] == "Vulkan" else ""
        self._start_chat()

    def _start_chat(self):
        system = npc_messages(self.npc, self.reasoning)
        history = list(system)
        llm = utils.load_llm(
            self.model,
            {"model_path": self.model["path"],
             "n_gpu_layers": -1 if self.device["type"] == "Vulkan" else 0,
             "n_ctx": 4096, "verbose": False},
            warmup_inputs=system + [{"role": "user", "content": "Hello."}],
            custom_jinja=True, reasoning=self.reasoning, log=True,
        )
        print(f"NPC: {self.npc['name']} | mode: {self.reasoning} | family: {self.model.get('family')}")
        print("Debug view streams RAW output (including plans). /exit to quit; /reset to clear history.")
        try:
            while True:
                try:
                    text = input("You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if text.lower() in ("/exit", "/quit"):
                    break
                if text.lower() == "/reset":
                    history = list(system)
                    llm.reset()
                    print("Conversation reset.")
                    continue
                if not text:
                    continue
                history.append({"role": "user", "content": text})
                print("Raw: ", end="", flush=True)
                result = generate(
                    llm, history, reasoning=self.reasoning, max_tokens=512,
                    temperature=0.7, seed=42, timeout=120,
                    on_text=lambda chunk: print(chunk, end="", flush=True),
                )
                print(f"\nStatus: {result.status} | TTFT {result.ttft:.3f}s | dialogue {result.dialogue_ttft:.3f}s")
                if result.error:
                    print(result.error)
                if result.status == "ok":
                    history.append({"role": "assistant", "content": result.raw})
                    if self.reasoning == "mini":
                        print(f"NPC dialogue: {result.dialogue}")
                else:
                    history.pop()
                    llm.reset()
                    print("Failed turn excluded from conversation. Try /reset or another prompt.")
        finally:
            llm.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning", choices=("off", "mini"), default="off")
    parser.add_argument("--npc")
    parser.add_argument("--model")
    parser.add_argument("--device", type=int)
    args = parser.parse_args()
    try:
        InteractiveChat(args.reasoning, args.npc, args.model, args.device)
    except (ValueError, OSError, utils.MyException) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
