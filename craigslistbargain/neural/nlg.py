"""Natural-language generation backends for dialogue-act surface realization."""

import hashlib
import json
import os
import random
import re
import string
import time
import urllib.error
import urllib.request

from nltk.tokenize import word_tokenize

from cocoa.core.entity import CanonicalEntity


DEFAULT_LLM_API_BASE = "https://api.openai.com/v1"

INTENT_GUIDELINES = {
    "greet": "Open the conversation or show lightweight interest. Do not close the deal.",
    "inquire": "Ask a question. Do not accept, reject, or close the deal.",
    "inform": "Provide information or answer a question. Do not close the deal.",
    "affirm": "Give a short yes/positive acknowledgement about a fact. Do not use deal-closing language.",
    "deny": "Give a short no/negative response or refusal. Do not sound accepting or willing.",
    "confirm": "Briefly confirm or clarify something. Do not close the deal.",
    "propose": "Propose the price PPRRIICCEE without final acceptance language.",
    "counter": "Make a counteroffer with PPRRIICCEE and make it clear negotiation is still ongoing.",
    "counter-noprice": "Continue negotiating without stating a new price. Do not say deal, accept, sold, see you soon, or pickup details.",
    "agree": "Explicitly agree to transact at PPRRIICCEE. Deal-closing language is allowed.",
    "agree-noprice": "Respond positively without stating a price, but avoid strong deal-closing language such as deal, sold, pickup, or see you soon.",
    "disagree": "Politely resist or reject the other side's position. Do not sound accepting.",
    "offer": "State the offer at PPRRIICCEE plainly.",
    "accept": "Accept briefly and clearly.",
    "reject": "Reject briefly and clearly.",
    "quit": "End the negotiation briefly.",
    "unknown": "Write a short neutral bargaining utterance.",
    "start": "Write a short opening utterance.",
}

SAFE_FALLBACKS = {
    "greet": "Hi, I am interested.",
    "inquire": "Can you tell me a little more about it?",
    "inform": "Thanks for the information.",
    "affirm": "Yes, that makes sense.",
    "deny": "No, I cannot do that.",
    "confirm": "Okay, just to confirm that is correct.",
    "propose": "Would you take PPRRIICCEE?",
    "counter": "I can do PPRRIICCEE, but that is my best offer.",
    "counter-noprice": "I am interested, but I would need you to come down a bit more.",
    "agree": "Okay, I can do PPRRIICCEE.",
    "agree-noprice": "Okay, that sounds reasonable.",
    "disagree": "I do not think that works for me.",
    "offer": "I can offer PPRRIICCEE.",
    "accept": "I accept.",
    "reject": "I cannot accept that.",
    "quit": "I will pass for now.",
    "unknown": "Okay.",
    "start": "Hi there.",
}


class BaseNLG(object):
    def _add_strategy_in_uttr(self, uttr, stra):
        uttr = uttr.copy()
        if random.randint(0, 5) > 0:
            return uttr
        uttr.insert(random.randint(0, len(uttr)), stra)
        return uttr

    def _tokens_from_template(self, template, lf, add_stra=None):
        words = word_tokenize(template)
        new_words = []
        for wd in words:
            if wd == "PPRRIICCEE" and lf.get("price") is not None:
                new_words.append(CanonicalEntity(type="price", value=lf.get("price")))
            else:
                new_words.append(wd)
        if add_stra is not None:
            new_words = self._add_strategy_in_uttr(new_words, add_stra)
        return new_words

    def _surface_from_template(self, template, lf, add_stra=None):
        words = word_tokenize(template)
        new_words = []
        for wd in words:
            if wd == "PPRRIICCEE" and lf.get("price") is not None:
                new_words.append("$" + str(lf.get("price")))
            else:
                new_words.append(wd)
        if add_stra is not None:
            new_words = self._add_strategy_in_uttr(new_words, add_stra)
        return "".join(
            [" " + tok if not tok.startswith("'") and tok not in string.punctuation else tok for tok in new_words]
        ).strip()


class IRNLG(BaseNLG):
    def __init__(self, args):
        self.gen_dic = {}
        with open(args.nlg_dir) as json_file:
            self.gen_dic = json.load(json_file)

    def gen(self, lf, role, category, as_tokens=False, add_stra=None):
        if self.gen_dic[category].get(lf.get("intent")) is None:
            new_words = [""]
            if add_stra is not None:
                new_words = self._add_strategy_in_uttr(new_words, add_stra)
            if not as_tokens:
                new_words = "".join(new_words)
            return new_words, (lf.get("intent"), role, category, 0)

        tid = random.randint(0, len(self.gen_dic[category][lf.get("intent")][role]) - 1)
        template = self.gen_dic[category][lf.get("intent")][role][tid]
        if as_tokens:
            return self._tokens_from_template(template, lf, add_stra=add_stra), (
                lf.get("intent"),
                role,
                category,
                tid,
            )
        return self._surface_from_template(template, lf, add_stra=add_stra), (
            lf.get("intent"),
            role,
            category,
            tid,
        )


class OpenAICompatibleNLG(BaseNLG):
    def __init__(self, args):
        self.model = getattr(args, "llm_nlg_model", None)
        self.api_base = getattr(args, "llm_nlg_api_base", DEFAULT_LLM_API_BASE).rstrip("/")
        self.api_key_env = getattr(args, "llm_nlg_api_key_env", "OPENAI_API_KEY")
        self.temperature = float(getattr(args, "llm_nlg_temperature", 0.2))
        self.timeout = int(getattr(args, "llm_nlg_timeout", 30))
        self.max_retries = int(getattr(args, "llm_nlg_max_retries", 2))
        self.cache_path = getattr(args, "llm_nlg_cache_path", os.path.join("cache", "llm_nlg_cache.json"))
        self.fallback_to_template = bool(getattr(args, "llm_nlg_fallback_to_template", False))
        self.template_backend = IRNLG(args) if self.fallback_to_template else None
        self.cache = self._load_cache()

        if not self.model:
            raise ValueError("`--llm-nlg-model` is required when `--nlg-backend llm` is used.")

    def _load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_cache(self):
        if not self.cache_path:
            return
        parent = os.path.dirname(self.cache_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as outfile:
            json.dump(self.cache, outfile, ensure_ascii=False, indent=2, sort_keys=True)

    def _request_key(self, lf, role, category, add_stra=None):
        key = {
            "intent": lf.get("intent"),
            "role": role,
            "category": category,
            "has_price": lf.get("price") is not None,
            "style_hint": add_stra,
        }
        return json.dumps(key, sort_keys=True, ensure_ascii=False)

    def _uid_from_key(self, key):
        return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]

    def _build_system_prompt(self):
        return (
            "You generate one short English Craigslist bargaining utterance from a dialogue act.\n"
            "Return only the utterance text, no quotes, no JSON, no speaker label.\n"
            "If a price should be mentioned, use the literal token PPRRIICCEE instead of any number.\n"
            "Keep it short, natural, and conversational.\n"
            "Do not invent product details.\n"
            "Do not use deal-closing language unless the act explicitly allows it."
        )

    def _build_user_prompt(self, lf, role, category, add_stra=None):
        intent = lf.get("intent")
        wants_price = lf.get("price") is not None
        style_hint = add_stra if add_stra is not None else "none"
        guideline = INTENT_GUIDELINES.get(intent, INTENT_GUIDELINES["unknown"])
        return (
            "Role: {role}\n"
            "Category: {category}\n"
            "Intent: {intent}\n"
            "Needs price mention: {needs_price}\n"
            "Strategy/style hint token: {style_hint}\n"
            "Intent guideline: {guideline}\n"
            "Write exactly one short utterance now."
        ).format(
            role=role,
            category=category,
            intent=intent,
            needs_price="yes" if wants_price else "no",
            style_hint=style_hint,
            guideline=guideline,
        )

    def _call_api(self, lf, role, category, add_stra=None):
        api_key = os.environ.get(self.api_key_env)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(lf, role, category, add_stra=add_stra)},
            ],
            "temperature": self.temperature,
            "max_tokens": 80,
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if api_key:
            req.add_header("Authorization", "Bearer " + api_key)

        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as err:
                last_err = err
                if attempt >= self.max_retries:
                    break
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError("LLM NLG request failed: {}".format(last_err))

    def _sanitize_generation(self, raw_text, wants_price):
        text = (raw_text or "").strip()
        if not text:
            return ""

        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
            text = re.sub(r"```$", "", text).strip()

        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    text = parsed.get("utterance", "") or parsed.get("text", "") or text
            except ValueError:
                pass

        text = text.splitlines()[0].strip()
        text = text.strip("\"' ")
        text = re.sub(r"\s+", " ", text)

        if wants_price and "PPRRIICCEE" not in text:
            return ""
        return text

    def _safe_fallback_template(self, lf):
        intent = lf.get("intent")
        template = SAFE_FALLBACKS.get(intent, SAFE_FALLBACKS["unknown"])
        if lf.get("price") is not None and "PPRRIICCEE" not in template:
            if intent in ("agree", "offer", "propose", "counter"):
                template = template.rstrip(".") + " PPRRIICCEE."
        return template

    def _template_for(self, lf, role, category, add_stra=None):
        key = self._request_key(lf, role, category, add_stra=add_stra)
        cached = self.cache.get(key)
        if cached:
            return cached, key

        template = ""
        try:
            raw = self._call_api(lf, role, category, add_stra=add_stra)
            template = self._sanitize_generation(raw, wants_price=lf.get("price") is not None)
        except RuntimeError:
            template = ""

        if not template and self.template_backend is not None:
            template, _ = self.template_backend.gen(lf, role, category, as_tokens=False, add_stra=None)
            if lf.get("price") is not None:
                template = template.replace("$" + str(lf.get("price")), "PPRRIICCEE")

        if not template:
            template = self._safe_fallback_template(lf)

        self.cache[key] = template
        self._save_cache()
        return template, key

    def gen(self, lf, role, category, as_tokens=False, add_stra=None):
        template, key = self._template_for(lf, role, category, add_stra=add_stra)
        uid = (lf.get("intent"), role, category, "llm", self._uid_from_key(key))
        if as_tokens:
            return self._tokens_from_template(template, lf, add_stra=add_stra), uid
        return self._surface_from_template(template, lf, add_stra=add_stra), uid


def build_nlg_module(args):
    backend = getattr(args, "nlg_backend", "template")
    if backend == "template":
        return IRNLG(args)
    if backend == "llm":
        return OpenAICompatibleNLG(args)
    raise ValueError("Unknown NLG backend: {}".format(backend))
