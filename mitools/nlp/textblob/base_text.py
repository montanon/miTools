import codecs
import os
import re
import string
import types
from itertools import chain
from pathlib import Path
from xml.etree import ElementTree

BaseString = (str, bytes)

SLASH, WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA = (
    "&slash;",
    "word",
    "part-of-speech",
    "chunk",
    "preposition",
    "relation",
    "anchor",
    "lemma",
)
UNIVERSAL = "universal"
NOUN, VERB, ADJ, ADV, PRON, DET, PREP, ADP, NUM, CONJ, INTJ, PRT, PUNC, X = (
    "NN",
    "VB",
    "JJ",
    "RB",
    "PR",
    "DT",
    "PP",
    "PP",
    "NO",
    "CJ",
    "UH",
    "PT",
    ".",
    "X",
)
TOKEN = re.compile(r"(\S+)\s")
PUNCTUATION = punctuation = ".,;:!?()[]{}`''\"@#$^&*+-|=~_"
ABBREVIATIONS = abbreviations = set(
    (
        "a.",
        "adj.",
        "adv.",
        "al.",
        "a.m.",
        "c.",
        "cf.",
        "comp.",
        "conf.",
        "def.",
        "ed.",
        "e.g.",
        "esp.",
        "etc.",
        "ex.",
        "f.",
        "fig.",
        "gen.",
        "id.",
        "i.e.",
        "int.",
        "l.",
        "m.",
        "Med.",
        "Mil.",
        "Mr.",
        "n.",
        "n.q.",
        "orig.",
        "pl.",
        "pred.",
        "pres.",
        "p.m.",
        "ref.",
        "v.",
        "vs.",
        "w/",
    )
)
RE_ABBR1 = re.compile(r"^[A-Za-z]\.$")  # single letter, "T. De Smedt"
RE_ABBR2 = re.compile(r"^([A-Za-z]\.)+$")  # alternating letters, "U.S."
RE_ABBR3 = re.compile(
    "^[A-Z]["
    + "|".join(  # capital followed by consonants, "Mr."
        "bcdfghjklmnpqrstvwxz"
    )
    + "]+.$"
)
EMOTICONS = {  # (facial expression, sentiment)-keys
    ("love", +1.00): set(("<3", "♥")),
    ("grin", +1.00): set(
        (">:D", ":-D", ":D", "=-D", "=D", "X-D", "x-D", "XD", "xD", "8-D")
    ),
    ("taunt", +0.75): set(
        (">:P", ":-P", ":P", ":-p", ":p", ":-b", ":b", ":c)", ":o)", ":^)")
    ),
    ("smile", +0.50): set(
        (">:)", ":-)", ":)", "=)", "=]", ":]", ":}", ":>", ":3", "8)", "8-)")
    ),
    ("wink", +0.25): set((">;]", ";-)", ";)", ";-]", ";]", ";D", ";^)", "*-)", "*)")),
    ("gasp", +0.05): set((">:o", ":-O", ":O", ":o", ":-o", "o_O", "o.O", "°O°", "°o°")),
    ("worry", -0.25): set(
        (">:/", ":-/", ":/", ":\\", ">:\\", ":-.", ":-s", ":s", ":S", ":-S", ">.>")
    ),
    ("frown", -0.75): set(
        (">:[", ":-(", ":(", "=(", ":-[", ":[", ":{", ":-<", ":c", ":-c", "=/")
    ),
    ("cry", -1.00): set((":'(", ":'''(", ";'(")),
}

RE_EMOTICONS = [
    r" ?".join([re.escape(each) for each in e]) for v in EMOTICONS.values() for e in v
]
RE_EMOTICONS = re.compile(r"(%s)($|\s)" % "|".join(RE_EMOTICONS))
RE_SARCASM = re.compile(r"\( ?\! ?\)")
replacements = {
    "'d": " 'd",
    "'m": " 'm",
    "'s": " 's",
    "'ll": " 'll",
    "'re": " 're",
    "'ve": " 've",
    "n't": " n't",
}
EOS = "END-OF-SENTENCE"
RE_ENTITY1 = re.compile(r"^http://")  # http://www.domain.com/path
RE_ENTITY2 = re.compile(r"^www\..*?\.[com|org|net|edu|de|uk]$")  # www.domain.com
RE_ENTITY3 = re.compile(r"^[\w\-\.\+]+@(\w[\w\-]+\.)+[\w\-]+$")  # name@domain.com
MOOD = "mood"  # emoticons, emojis
IRONY = "irony"  # sarcasm mark (!)
NOUN, VERB, ADJECTIVE, ADVERB = "NN", "VB", "JJ", "RB"
RE_SYNSET = re.compile(r"^[acdnrv][-_][0-9]+$")
CD = re.compile(r"^[0-9\-\,\.\:\/\%\$]+$")
SEPARATOR = "/"

NN = r"NN|NNS|NNP|NNPS|NNPS?\-[A-Z]{3,4}|PR|PRP|PRP\$"
VB = r"VB|VBD|VBG|VBN|VBP|VBZ"
JJ = r"JJ|JJR|JJS"
RB = r"(?<!W)RB|RBR|RBS"
CHUNKS = [
    [
        # Germanic languages: en, de, nl, ...
        (
            "NP",
            re.compile(
                r"(("
                + NN
                + ")/)*((DT|CD|CC|CJ)/)*(("
                + RB
                + "|"
                + JJ
                + ")/)*(("
                + NN
                + ")/)+"
            ),
        ),
        ("VP", re.compile(r"(((MD|" + RB + ")/)*((" + VB + ")/)+)+")),
        ("VP", re.compile(r"((MD)/)")),
        ("PP", re.compile(r"((IN|PP|TO)/)+")),
        ("ADJP", re.compile(r"((CC|CJ|" + RB + "|" + JJ + ")/)*((" + JJ + ")/)+")),
        ("ADVP", re.compile(r"((" + RB + "|WRB)/)+")),
    ],
    [
        # Romance languages: es, fr, it, ...
        (
            "NP",
            re.compile(
                r"(("
                + NN
                + ")/)*((DT|CD|CC|CJ)/)*(("
                + RB
                + "|"
                + JJ
                + ")/)*(("
                + NN
                + ")/)+(("
                + RB
                + "|"
                + JJ
                + ")/)*"
            ),
        ),
        ("VP", re.compile(r"(((MD|" + RB + ")/)*((" + VB + ")/)+((" + RB + ")/)*)+")),
        ("VP", re.compile(r"((MD)/)")),
        ("PP", re.compile(r"((IN|PP|TO)/)+")),
        ("ADJP", re.compile(r"((CC|CJ|" + RB + "|" + JJ + ")/)*((" + JJ + ")/)+")),
        ("ADVP", re.compile(r"((" + RB + "|WRB)/)+")),
    ],
]

# Handle ADJP before VP, so that
# RB prefers next ADJP over previous VP.
CHUNKS[0].insert(1, CHUNKS[0].pop(3))
CHUNKS[1].insert(1, CHUNKS[1].pop(3))
PTB = PENN = "penn"
TOKENS = "tokens"


def decode_string(value, encoding="utf-8"):
    if isinstance(encoding, BaseString):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(value, bytes):
        for e in encoding:
            try:
                return value.decode(*e)
            except Exception:
                pass
        return value
    return str(value)


def encode_string(value, encoding="utf-8"):
    if isinstance(encoding, BaseString):
        encoding = ((encoding,),) + (("windows-1252",), ("utf-8", "ignore"))
    if isinstance(value, str):
        for e in encoding:
            try:
                return value.encode(*e)
            except Exception:
                pass
        return value
    return str(value)


decode_utf8 = decode_string
encode_utf8 = encode_string


def isnumeric(strg):
    try:
        float(strg)
    except ValueError:
        return False
    return True


class lazydict(dict):
    def load(self):
        pass

    def _lazy(self, method, *args):
        if dict.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(dict, method), self))
        return getattr(dict, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")

    def __len__(self):
        return self._lazy("__len__")

    def __iter__(self):
        return self._lazy("__iter__")

    def __contains__(self, *args):
        return self._lazy("__contains__", *args)

    def __getitem__(self, *args):
        return self._lazy("__getitem__", *args)

    def __setitem__(self, *args):
        return self._lazy("__setitem__", *args)

    def setdefault(self, *args):
        return self._lazy("setdefault", *args)

    def get(self, *args, **kwargs):
        return self._lazy("get", *args)

    def items(self):
        return self._lazy("items")

    def keys(self):
        return self._lazy("keys")

    def values(self):
        return self._lazy("values")

    def update(self, *args):
        return self._lazy("update", *args)

    def pop(self, *args):
        return self._lazy("pop", *args)

    def popitem(self, *args):
        return self._lazy("popitem", *args)


class lazylist(list):
    def load(self):
        pass

    def _lazy(self, method, *args):
        if list.__len__(self) == 0:
            self.load()
            setattr(self, method, types.MethodType(getattr(list, method), self))
        return getattr(list, method)(self, *args)

    def __repr__(self):
        return self._lazy("__repr__")

    def __len__(self):
        return self._lazy("__len__")

    def __iter__(self):
        return self._lazy("__iter__")

    def __contains__(self, *args):
        return self._lazy("__contains__", *args)

    def insert(self, *args):
        return self._lazy("insert", *args)

    def append(self, *args):
        return self._lazy("append", *args)

    def extend(self, *args):
        return self._lazy("extend", *args)

    def remove(self, *args):
        return self._lazy("remove", *args)

    def pop(self, *args):
        return self._lazy("pop", *args)


def penntreebank_to_universal(token, tag):
    if tag.startswith(("NNP-", "NNPS-")):
        return (token, "{}-{}".format(NOUN, tag.split("-")[-1]))
    if tag in ("NN", "NNS", "NNP", "NNPS", "NP"):
        return (token, NOUN)
    if tag in ("MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return (token, VERB)
    if tag in ("JJ", "JJR", "JJS"):
        return (token, ADJ)
    if tag in ("RB", "RBR", "RBS", "WRB"):
        return (token, ADV)
    if tag in ("PRP", "PRP$", "WP", "WP$"):
        return (token, PRON)
    if tag in ("DT", "PDT", "WDT", "EX"):
        return (token, DET)
    if tag in ("IN",):
        return (token, PREP)
    if tag in ("CD",):
        return (token, NUM)
    if tag in ("CC",):
        return (token, CONJ)
    if tag in ("UH",):
        return (token, INTJ)
    if tag in ("POS", "RP", "TO"):
        return (token, PRT)
    if tag in ("SYM", "LS", ".", "!", "?", ",", ":", "(", ")", '"', "#", "$"):
        return (token, PUNC)
    return (token, X)


def find_tokens(
    text: str,
    punctuation=PUNCTUATION,
    abbreviations=ABBREVIATIONS,
    replace=replacements,
    linebreak=r"\n{2,}",
):
    punctuation = tuple(punctuation.replace(".", ""))
    for a, b in list(replace.items()):
        text = re.sub(a, b, text)
    if isinstance(text, str):
        text = (
            str(text)
            .replace("“", " “ ")
            .replace("”", " ” ")
            .replace("‘", " ‘ ")
            .replace("’", " ’ ")
            .replace("'", " ' ")
            .replace('"', ' " ')
        )
    text = re.sub("\r\n", "\n", text)
    text = re.sub(linebreak, " %s " % EOS, text)
    text = re.sub(r"\s+", " ", text)
    tokens = []
    for t in TOKEN.findall(text + " "):
        if len(t) > 0:
            tail = []
            while t.startswith(punctuation) and t not in replace:
                if t.startswith(punctuation):
                    tokens.append(t[0])
                    t = t[1:]
            while t.endswith(punctuation + (".",)) and t not in replace:
                if t.endswith(punctuation):
                    tail.append(t[-1])
                    t = t[:-1]
                if t.endswith("..."):
                    tail.append("...")
                    t = t[:-3].rstrip(".")
                if t.endswith("."):
                    if (
                        t in abbreviations
                        or RE_ABBR1.match(t) is not None
                        or RE_ABBR2.match(t) is not None
                        or RE_ABBR3.match(t) is not None
                    ):
                        break
                    else:
                        tail.append(t[-1])
                        t = t[:-1]
            if t != "":
                tokens.append(t)
            tokens.extend(reversed(tail))
    sentences, i, j = [[]], 0, 0
    while j < len(tokens):
        if tokens[j] in ("...", ".", "!", "?", EOS):
            while j < len(tokens) and tokens[j] in (
                "'",
                '"',
                "”",
                "’",
                "...",
                ".",
                "!",
                "?",
                ")",
                EOS,
            ):
                if tokens[j] in ("'", '"') and sentences[-1].count(tokens[j]) % 2 == 0:
                    break  # Balanced quotes.
                j += 1
            sentences[-1].extend(t for t in tokens[i:j] if t != EOS)
            sentences.append([])
            i = j
        j += 1
    sentences[-1].extend(tokens[i:j])
    sentences = (" ".join(s) for s in sentences if len(s) > 0)
    sentences = (RE_SARCASM.sub("(!)", s) for s in sentences)
    sentences = [
        RE_EMOTICONS.sub(lambda m: m.group(1).replace(" ", "") + m.group(2), s)
        for s in sentences
    ]
    return sentences


def _read(path: Path, encoding="utf-8", comment=";;;"):
    if path:
        if isinstance(path, BaseString) and path.exists():
            f = open(path, encoding=encoding)
        elif isinstance(path, BaseString):
            f = path.splitlines()
        elif hasattr(path, "read"):
            f = path.read().splitlines()
        else:
            f = path
        for i, line in enumerate(f):
            line = (
                line.strip(codecs.BOM_UTF8)
                if i == 0 and isinstance(line, bytes)
                else line
            )
            line = line.strip()
            line = decode_utf8(line)
            if not line or (comment and line.startswith(comment)):
                continue
            yield line
    return


class Lexicon(lazydict):
    def __init__(
        self,
        path="",
        morphology=None,
        context=None,
        entities=None,
        NNP="NNP",
        language=None,
    ):
        self._path = path
        self._language = language
        self.morphology = Morphology(self, path=morphology)
        self.context = Context(self, path=context)
        self.entities = Entities(self, path=entities, tag=NNP)

    def load(self):
        dict.update(self, (x.split(" ")[:2] for x in _read(self._path) if x.strip()))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language


class Rules:
    def __init__(self, lexicon=None, cmd=None):
        if cmd is None:
            cmd = {}
        if lexicon is None:
            lexicon = {}
        self.lexicon, self.cmd = lexicon, cmd

    def apply(self, x):
        return x


class Morphology(lazylist, Rules):
    def __init__(self, lexicon=None, path=""):
        if lexicon is None:
            lexicon = {}
        cmd = (
            "char",  # Word contains x.
            "haspref",  # Word starts with x.
            "hassuf",  # Word end with x.
            "addpref",  # x + word is in lexicon.
            "addsuf",  # Word + x is in lexicon.
            "deletepref",  # Word without x at the start is in lexicon.
            "deletesuf",  # Word without x at the end is in lexicon.
            "goodleft",  # Word preceded by word x.
            "goodright",  # Word followed by word x.
        )
        cmd = dict.fromkeys(cmd, True)
        cmd.update(("f" + k, v) for k, v in list(cmd.items()))
        Rules.__init__(self, lexicon, cmd)
        self._path = path

    @property
    def path(self):
        return self._path

    def load(self):
        list.extend(self, (x.split() for x in _read(self._path)))

    def apply(self, token, previous=(None, None), next=(None, None)):
        w = token[0]
        for r in self:
            if r[1] in self.cmd:  # Rule = ly hassuf 2 RB x
                f, x, pos, cmd = bool(0), r[0], r[-2], r[1].lower()
            if r[2] in self.cmd:  # Rule = NN s fhassuf 1 NNS x
                f, x, pos, cmd = bool(1), r[1], r[-2], r[2].lower().lstrip("f")
            if f and token[1] != r[0]:
                continue
            if (
                (cmd == "char" and x in w)
                or (cmd == "haspref" and w.startswith(x))
                or (cmd == "hassuf" and w.endswith(x))
                or (cmd == "addpref" and x + w in self.lexicon)
                or (cmd == "addsuf" and w + x in self.lexicon)
                or (
                    cmd == "deletepref"
                    and w.startswith(x)
                    and w[len(x) :] in self.lexicon
                )
                or (
                    cmd == "deletesuf"
                    and w.endswith(x)
                    and w[: -len(x)] in self.lexicon
                )
                or (cmd == "goodleft" and x == next[0])
                or (cmd == "goodright" and x == previous[0])
            ):
                token[1] = pos
        return token

    def insert(self, i, tag, affix, cmd="hassuf", tagged=None):
        if affix.startswith("-") and affix.endswith("-"):
            affix, cmd = affix[+1:-1], "char"
        if affix.startswith("-"):
            affix, cmd = affix[+1:-0], "hassuf"
        if affix.endswith("-"):
            affix, cmd = affix[+0:-1], "haspref"
        if tagged:
            r = [tagged, affix, "f" + cmd.lstrip("f"), tag, "x"]
        else:
            r = [affix, cmd.lstrip("f"), tag, "x"]
        lazylist.insert(self, i, r)

    def append(self, *args, **kwargs):
        self.insert(len(self) - 1, *args, **kwargs)

    def extend(self, rules=None):
        if rules is None:
            rules = []
        for r in rules:
            self.append(*r)


class Context(lazylist, Rules):
    def __init__(self, lexicon=None, path=""):
        if lexicon is None:
            lexicon = {}
        cmd = (
            "prevtag",  # Preceding word is tagged x.
            "nexttag",  # Following word is tagged x.
            "prev2tag",  # Word 2 before is tagged x.
            "next2tag",  # Word 2 after is tagged x.
            "prev1or2tag",  # One of 2 preceding words is tagged x.
            "next1or2tag",  # One of 2 following words is tagged x.
            "prev1or2or3tag",  # One of 3 preceding words is tagged x.
            "next1or2or3tag",  # One of 3 following words is tagged x.
            "surroundtag",  # Preceding word is tagged x and following word is tagged y.
            "curwd",  # Current word is x.
            "prevwd",  # Preceding word is x.
            "nextwd",  # Following word is x.
            "prev1or2wd",  # One of 2 preceding words is x.
            "next1or2wd",  # One of 2 following words is x.
            "next1or2or3wd",  # One of 3 preceding words is x.
            "prev1or2or3wd",  # One of 3 following words is x.
            "prevwdtag",  # Preceding word is x and tagged y.
            "nextwdtag",  # Following word is x and tagged y.
            "wdprevtag",  # Current word is y and preceding word is tagged x.
            "wdnexttag",  # Current word is x and following word is tagged y.
            "wdand2aft",  # Current word is x and word 2 after is y.
            "wdand2tagbfr",  # Current word is y and word 2 before is tagged x.
            "wdand2tagaft",  # Current word is x and word 2 after is tagged y.
            "lbigram",  # Current word is y and word before is x.
            "rbigram",  # Current word is x and word after is y.
            "prevbigram",  # Preceding word is tagged x and word before is tagged y.
            "nextbigram",  # Following word is tagged x and word after is tagged y.
        )
        Rules.__init__(self, lexicon, dict.fromkeys(cmd, True))
        self._path = path

    @property
    def path(self):
        return self._path

    def load(self):
        # ["VBD", "VB", "PREVTAG", "TO"]
        list.extend(self, (x.split() for x in _read(self._path)))

    def apply(self, tokens):
        o = [("STAART", "STAART")] * 3  # Empty delimiters for look ahead/back.
        t = o + tokens + o
        for i, token in enumerate(t):
            for r in self:
                if token[1] == "STAART":
                    continue
                if token[1] != r[0] and r[0] != "*":
                    continue
                cmd, x, y = r[2], r[3], r[4] if len(r) > 4 else ""
                cmd = cmd.lower()
                if (
                    (cmd == "prevtag" and x == t[i - 1][1])
                    or (cmd == "nexttag" and x == t[i + 1][1])
                    or (cmd == "prev2tag" and x == t[i - 2][1])
                    or (cmd == "next2tag" and x == t[i + 2][1])
                    or (cmd == "prev1or2tag" and x in (t[i - 1][1], t[i - 2][1]))
                    or (cmd == "next1or2tag" and x in (t[i + 1][1], t[i + 2][1]))
                    or (
                        cmd == "prev1or2or3tag"
                        and x in (t[i - 1][1], t[i - 2][1], t[i - 3][1])
                    )
                    or (
                        cmd == "next1or2or3tag"
                        and x in (t[i + 1][1], t[i + 2][1], t[i + 3][1])
                    )
                    or (cmd == "surroundtag" and x == t[i - 1][1] and y == t[i + 1][1])
                    or (cmd == "curwd" and x == t[i + 0][0])
                    or (cmd == "prevwd" and x == t[i - 1][0])
                    or (cmd == "nextwd" and x == t[i + 1][0])
                    or (cmd == "prev1or2wd" and x in (t[i - 1][0], t[i - 2][0]))
                    or (cmd == "next1or2wd" and x in (t[i + 1][0], t[i + 2][0]))
                    or (cmd == "prevwdtag" and x == t[i - 1][0] and y == t[i - 1][1])
                    or (cmd == "nextwdtag" and x == t[i + 1][0] and y == t[i + 1][1])
                    or (cmd == "wdprevtag" and x == t[i - 1][1] and y == t[i + 0][0])
                    or (cmd == "wdnexttag" and x == t[i + 0][0] and y == t[i + 1][1])
                    or (cmd == "wdand2aft" and x == t[i + 0][0] and y == t[i + 2][0])
                    or (cmd == "wdand2tagbfr" and x == t[i - 2][1] and y == t[i + 0][0])
                    or (cmd == "wdand2tagaft" and x == t[i + 0][0] and y == t[i + 2][1])
                    or (cmd == "lbigram" and x == t[i - 1][0] and y == t[i + 0][0])
                    or (cmd == "rbigram" and x == t[i + 0][0] and y == t[i + 1][0])
                    or (cmd == "prevbigram" and x == t[i - 2][1] and y == t[i - 1][1])
                    or (cmd == "nextbigram" and x == t[i + 1][1] and y == t[i + 2][1])
                ):
                    t[i] = [t[i][0], r[1]]
        return t[len(o) : -len(o)]

    def insert(self, i, tag1, tag2, cmd="prevtag", x=None, y=None):
        if " < " in tag1 and not x and not y:
            tag1, x = tag1.split(" < ")
            cmd = "prevtag"
        if " > " in tag1 and not x and not y:
            x, tag1 = tag1.split(" > ")
            cmd = "nexttag"
        lazylist.insert(self, i, [tag1, tag2, cmd, x or "", y or ""])

    def append(self, *args, **kwargs):
        self.insert(len(self) - 1, *args, **kwargs)

    def extend(self, rules=None):
        if rules is None:
            rules = []
        for r in rules:
            self.append(*r)


class Entities(lazydict, Rules):
    def __init__(self, lexicon=None, path="", tag="NNP"):
        if lexicon is None:
            lexicon = {}
        cmd = (
            "pers",  # Persons: George/NNP-PERS
            "loc",  # Locations: Washington/NNP-LOC
            "org",  # Organizations: Google/NNP-ORG
        )
        Rules.__init__(self, lexicon, cmd)
        self._path = path
        self.tag = tag

    @property
    def path(self):
        return self._path

    def load(self):
        for x in _read(self.path):
            x = [x.lower() for x in x.split()]
            dict.setdefault(self, x[0], []).append(x)

    def apply(self, tokens):
        i = 0
        while i < len(tokens):
            w = tokens[i][0].lower()
            if RE_ENTITY1.match(w) or RE_ENTITY2.match(w) or RE_ENTITY3.match(w):
                tokens[i][1] = self.tag
            if w in self:
                for e in self[w]:
                    e, tag = (
                        (e[:-1], "-" + e[-1].upper()) if e[-1] in self.cmd else (e, "")
                    )
                    b = True
                    for j, e in enumerate(e):
                        if i + j >= len(tokens) or tokens[i + j][0].lower() != e:
                            b = False
                            break
                    if b:
                        for token in tokens[i : i + j + 1]:
                            token[1] = (
                                token[1] == "NNPS" and token[1] or self.tag
                            ) + tag
                        i += j
                        break
            i += 1
        return tokens

    def append(self, entity, name="pers"):
        e = [s.lower() for s in entity.split(" ") + [name]]
        self.setdefault(e[0], []).append(e)

    def extend(self, entities):
        for entity, name in entities:
            self.append(entity, name)


def avg(list):
    return sum(list) / float(len(list) or 1)


class Score(tuple):
    def __new__(self, polarity, subjectivity, assessments=None):
        if assessments is None:
            assessments = []
        return tuple.__new__(self, [polarity, subjectivity])

    def __init__(self, polarity, subjectivity, assessments=None):
        if assessments is None:
            assessments = []
        self.assessments = assessments


class Sentiment(lazydict):
    def __init__(self, path="", language=None, synset=None, confidence=None, **kwargs):
        self._path = path  # XML file path.
        self._language = None  # XML language attribute ("en", "fr", ...)
        self._confidence = None  # XML confidence attribute threshold (>=).
        self._synset = synset  # XML synset attribute ("wordnet_id", "cornetto_id", ...)
        self._synsets = {}  # {"a-01123879": (1.0, 1.0, 1.0)}
        self.labeler = {}  # {"dammit": "profanity"}
        self.tokenizer = kwargs.get("tokenizer", find_tokens)
        self.negations = kwargs.get("negations", ("no", "not", "n't", "never"))
        self.modifiers = kwargs.get("modifiers", ("RB",))
        self.modifier = kwargs.get("modifier", lambda w: w.endswith("ly"))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @property
    def confidence(self):
        return self._confidence

    def load(self, path=None):
        if not path:
            path = self._path
        if not os.path.exists(path):
            return
        words, synsets, labels = {}, {}, {}
        xml = ElementTree.parse(path)
        xml = xml.getroot()
        for w in xml.findall("word"):
            if self._confidence is None or self._confidence <= float(
                w.attrib.get("confidence", 0.0)
            ):
                w, pos, p, s, i, label, synset = (
                    w.attrib.get("form"),
                    w.attrib.get("pos"),
                    w.attrib.get("polarity", 0.0),
                    w.attrib.get("subjectivity", 0.0),
                    w.attrib.get("intensity", 1.0),
                    w.attrib.get("label"),
                    w.attrib.get(self._synset),  # wordnet_id, cornetto_id, ...
                )
                psi = (float(p), float(s), float(i))
                if w:
                    words.setdefault(w, {}).setdefault(pos, []).append(psi)
                if w and label:
                    labels[w] = label
                if synset:
                    synsets.setdefault(synset, []).append(psi)
        self._language = xml.attrib.get("language", self._language)
        for w in words:
            words[w] = dict(
                (pos, [avg(each) for each in zip(*psi)])
                for pos, psi in words[w].items()
            )
        for w, pos in list(words.items()):
            words[w][None] = [avg(each) for each in zip(*pos.values())]
        for id, psi in synsets.items():
            synsets[id] = [avg(each) for each in zip(*psi)]
        dict.update(self, words)
        dict.update(self.labeler, labels)
        dict.update(self._synsets, synsets)

    def synset(self, id, pos=ADJECTIVE):
        id = str(id).zfill(8)
        if not id.startswith(("n-", "v-", "a-", "r-")):
            if pos == NOUN:
                id = "n-" + id
            if pos == VERB:
                id = "v-" + id
            if pos == ADJECTIVE:
                id = "a-" + id
            if pos == ADVERB:
                id = "r-" + id
        if dict.__len__(self) == 0:
            self.load()
        return tuple(self._synsets.get(id, (0.0, 0.0))[:2])

    def __call__(self, s, negation=True, **kwargs):
        def avg(assessments, weighted=lambda w: 1):
            s, n = 0, 0
            for words, score in assessments:
                w = weighted(words)
                s += w * score
                n += w
            return s / float(n or 1)

        if hasattr(s, "gloss"):
            a = [(s.synonyms[0],) + self.synset(s.id, pos=s.pos) + (None,)]
        elif (
            isinstance(s, BaseString) and RE_SYNSET.match(s) and hasattr(s, "synonyms")
        ):
            a = [(s.synonyms[0],) + self.synset(s.id, pos=s.pos) + (None,)]
        elif isinstance(s, BaseString):
            a = self.assessments(
                ((w.lower(), None) for w in " ".join(self.tokenizer(s)).split()),
                negation,
            )
        elif hasattr(s, "sentences"):
            a = self.assessments(
                (
                    (w.lemma or w.string.lower(), w.pos[:2])
                    for w in chain.from_iterable(s)
                ),
                negation,
            )
        elif hasattr(s, "lemmata"):
            a = self.assessments(
                ((w.lemma or w.string.lower(), w.pos[:2]) for w in s.words), negation
            )
        elif hasattr(s, "lemma"):
            a = self.assessments(((s.lemma or s.string.lower(), s.pos[:2]),), negation)
        elif hasattr(s, "terms"):
            a = self.assessments(
                chain.from_iterable(((w, None), (None, None)) for w in s), negation
            )
            kwargs.setdefault("weight", lambda w: s.terms[w[0]])
        elif isinstance(s, dict):
            a = self.assessments(
                chain.from_iterable(((w, None), (None, None)) for w in s), negation
            )
            kwargs.setdefault("weight", lambda w: s[w[0]])
        elif isinstance(s, list):
            a = self.assessments(((w, None) for w in s), negation)
        else:
            a = []
        weight = kwargs.get("weight", lambda w: 1)  # [(w, p) for w, p, s, x in a]
        return Score(
            polarity=avg([(w, p) for w, p, s, x in a], weight),
            subjectivity=avg([(w, s) for w, p, s, x in a], weight),
            assessments=a,
        )

    def assessments(self, words=None, negation=True):
        if words is None:
            words = []
        a = []
        m = None  # Preceding modifier (i.e., adverb or adjective).
        n = None  # Preceding negation (e.g., "not beautiful").
        for w, pos in words:
            if w is None:
                continue
            if w in self and pos in self[w]:
                p, s, i = self[w][pos]
                if m is None:
                    a.append(dict(w=[w], p=p, s=s, i=i, n=1, x=self.labeler.get(w)))
                if m is not None:
                    a[-1]["w"].append(w)
                    a[-1]["p"] = max(-1.0, min(p * a[-1]["i"], +1.0))
                    a[-1]["s"] = max(-1.0, min(s * a[-1]["i"], +1.0))
                    a[-1]["i"] = i
                    a[-1]["x"] = self.labeler.get(w)
                if n is not None:
                    a[-1]["w"].insert(0, n)
                    a[-1]["i"] = 1.0 / a[-1]["i"]
                    a[-1]["n"] = -1
                m = None
                n = None
                if (
                    pos
                    and pos in self.modifiers
                    or any(map(self[w].__contains__, self.modifiers))
                ):
                    m = (w, pos)
                if negation and w in self.negations:
                    n = w
            else:
                if negation and w in self.negations:
                    n = w
                elif n and len(w.strip("'")) > 1:
                    n = None
                if (
                    n is not None
                    and m is not None
                    and (pos in self.modifiers or self.modifier(m[0]))
                ):
                    a[-1]["w"].append(n)
                    a[-1]["n"] = -1
                    n = None
                elif m and len(w) > 2:
                    m = None
                if w == "!" and len(a) > 0:
                    a[-1]["w"].append("!")
                    a[-1]["p"] = max(-1.0, min(a[-1]["p"] * 1.25, +1.0))
                if w == "(!)":
                    a.append(dict(w=[w], p=0.0, s=1.0, i=1.0, n=1, x=IRONY))
                if w.isalpha() is False and len(w) <= 5 and w not in PUNCTUATION:
                    for (_type, p), e in EMOTICONS.items():
                        if w in map(lambda e: e.lower(), e):
                            a.append(dict(w=[w], p=p, s=1.0, i=1.0, n=1, x=MOOD))
                            break
        for i in range(len(a)):
            w = a[i]["w"]
            p = a[i]["p"]
            s = a[i]["s"]
            n = a[i]["n"]
            x = a[i]["x"]
            a[i] = (w, p * -0.5 if n < 0 else p, s, x)
        return a

    def annotate(
        self, word, pos=None, polarity=0.0, subjectivity=0.0, intensity=1.0, label=None
    ):
        w = self.setdefault(word, {})
        w[pos] = w[None] = (polarity, subjectivity, intensity)
        if label:
            self.labeler[word] = label


def suffix_rules(token, tag="NN"):
    if isinstance(token, (list, tuple)):
        token, tag = token
    if token.endswith("ing"):
        tag = "VBG"
    if token.endswith("ly"):
        tag = "RB"
    if token.endswith("s") and not token.endswith(("is", "ous", "ss")):
        tag = "NNS"
    if (
        token.endswith(
            ("able", "al", "ful", "ible", "ient", "ish", "ive", "less", "tic", "ous")
        )
        or "-" in token
    ):
        tag = "JJ"
    if token.endswith("ed"):
        tag = "VBN"
    if token.endswith(("ate", "ify", "ise", "ize")):
        tag = "VBP"
    return [token, tag]


def find_tags(
    tokens,
    lexicon=None,
    model=None,
    morphology=None,
    context=None,
    entities=None,
    default=("NN", "NNP", "CD"),
    language="en",
    map=None,
    **kwargs,
):
    if lexicon is None:
        lexicon = {}
    tagged = []
    for i, token in enumerate(tokens):
        tagged.append(
            [token, lexicon.get(token, i == 0 and lexicon.get(token.lower()) or None)]
        )
    for i, (token, tag) in enumerate(tagged):
        prev, next = (None, None), (None, None)
        if i > 0:
            prev = tagged[i - 1]
        if i < len(tagged) - 1:
            next = tagged[i + 1]
        if tag is None or token in (model is not None and model.unknown or ()):
            if model is not None:
                tagged[i] = model.apply([token, None], prev, next)
            elif token.istitle() and language != "de":
                tagged[i] = [token, default[1]]
            elif CD.match(token) is not None:
                tagged[i] = [token, default[2]]
            elif morphology is not None:
                tagged[i] = morphology.apply([token, default[0]], prev, next)
            elif language == "en":
                tagged[i] = suffix_rules([token, default[0]])
            else:
                tagged[i] = [token, default[0]]
    if context is not None and model is None:
        tagged = context.apply(tagged)
    if entities is not None:
        tagged = entities.apply(tagged)
    if map is not None:
        tagged = [list(map(token, tag)) or [token, default[0]] for token, tag in tagged]
    return tagged


def find_chunks(tagged, language="en"):
    chunked = [x for x in tagged]
    tags = "".join(f"{tag}{SEPARATOR}" for token, tag in tagged)
    for tag, rule in CHUNKS[
        int(language in ("ca", "es", "pt", "fr", "it", "pt", "ro"))
    ]:
        for m in rule.finditer(tags):
            i = m.start()
            j = tags[:i].count(SEPARATOR)
            n = m.group(0).count(SEPARATOR)
            for k in range(j, j + n):
                if len(chunked[k]) == 3:
                    continue
                if len(chunked[k]) < 3:
                    if k == j and chunked[k][1] in ("CC", "CJ", "KON", "Conj(neven)"):
                        j += 1
                    elif k == j:
                        chunked[k].append("B-" + tag)
                    else:
                        chunked[k].append("I-" + tag)
    for chink in filter(lambda x: len(x) < 3, chunked):
        chink.append("O")
    for i, (_word, tag, chunk) in enumerate(chunked):
        if tag.startswith("RB") and chunk == "B-NP":
            if i < len(chunked) - 1 and not chunked[i + 1][1].startswith("JJ"):
                chunked[i + 0][2] = "B-ADVP"
                chunked[i + 1][2] = "B-NP"
    return chunked


def find_prepositions(chunked):
    for ch in chunked:
        ch.append("O")
    for i, chunk in enumerate(chunked):
        if chunk[2].endswith("PP") and chunk[-1] == "O":
            if i < len(chunked) - 1 and (
                chunked[i + 1][2].endswith(("NP", "PP"))
                or chunked[i + 1][1] in ("VBG", "VBN")
            ):
                chunk[-1] = "B-PNP"
                pp = True
                for ch in chunked[i + 1 :]:
                    if not (ch[2].endswith(("NP", "PP")) or ch[1] in ("VBG", "VBN")):
                        break
                    if ch[2].endswith("PP") and pp:
                        ch[-1] = "I-PNP"
                    if not ch[2].endswith("PP"):
                        ch[-1] = "I-PNP"
                        pp = False
    return chunked


class Parser:
    def __init__(self, lexicon=None, default=("NN", "NNP", "CD"), language=None):
        if lexicon is None:
            lexicon = {}
        self.lexicon = lexicon
        self.default = default
        self.language = language

    def find_tokens(self, string, **kwargs):
        return find_tokens(
            str(string),
            punctuation=kwargs.get("punctuation", PUNCTUATION),
            abbreviations=kwargs.get("abbreviations", ABBREVIATIONS),
            replace=kwargs.get("replace", replacements),
            linebreak=r"\n{2,}",
        )

    def find_tags(self, tokens, **kwargs):
        return find_tags(
            tokens,
            language=kwargs.get("language", self.language),
            lexicon=kwargs.get("lexicon", self.lexicon),
            default=kwargs.get("default", self.default),
            map=kwargs.get("map", None),
        )

    def find_chunks(self, tokens, **kwargs):
        return find_prepositions(
            find_chunks(tokens, language=kwargs.get("language", self.language))
        )

    def find_prepositions(self, tokens, **kwargs):
        return find_prepositions(tokens)  # See also Parser.find_chunks().

    def find_labels(self, tokens, **kwargs):
        return find_relations(tokens)

    def find_lemmata(self, tokens, **kwargs):
        return [token + [token[0].lower()] for token in tokens]

    def parse(
        self,
        s,
        tokenize=True,
        tags=True,
        chunks=True,
        relations=False,
        lemmata=False,
        encoding="utf-8",
        **kwargs,
    ):
        if tokenize:
            s = self.find_tokens(s, **kwargs)
        if isinstance(s, (list, tuple)):
            s = [isinstance(s, BaseString) and s.split(" ") or s for s in s]
        if isinstance(s, BaseString):
            s = [s.split(" ") for s in s.split("\n")]
        for i in range(len(s)):
            for j in range(len(s[i])):
                if isinstance(s[i][j], bytes):
                    s[i][j] = decode_string(s[i][j], encoding)
            if tags or chunks or relations or lemmata:
                s[i] = self.find_tags(s[i], **kwargs)
            else:
                s[i] = [[w] for w in s[i]]
            if chunks or relations:
                s[i] = self.find_chunks(s[i], **kwargs)
            if relations:
                s[i] = self.find_labels(s[i], **kwargs)
            if lemmata:
                s[i] = self.find_lemmata(s[i], **kwargs)
        if not kwargs.get("collapse", True) or kwargs.get("split", False):
            return s
        format = ["word"]
        if tags:
            format.append("part-of-speech")
        if chunks:
            format.extend(("chunk", "preposition"))
        if relations:
            format.append("relation")
        if lemmata:
            format.append("lemma")
        for i in range(len(s)):
            for j in range(len(s[i])):
                s[i][j][0] = s[i][j][0].replace("/", "&slash;")
                s[i][j] = "/".join(s[i][j])
            s[i] = " ".join(s[i])
        s = "\n".join(s)
        s = TaggedString(str(s), format, language=kwargs.get("language", self.language))
        return s


class TaggedString(str):
    def __new__(self, string, tags=None, language=None):
        if tags is None:
            tags = ["word"]
        if isinstance(string, str) and hasattr(string, "tags"):
            tags, language = string.tags, string.language
        if isinstance(string, list):
            string = [
                [[x.replace("/", "&slash;") for x in token] for token in s]
                for s in string
            ]
            string = "\n".join(" ".join("/".join(token) for token in s) for s in string)
        s = str.__new__(self, string)
        s.tags = list(tags)
        s.language = language
        return s

    def split(self, sep=TOKENS):
        if sep != TOKENS:
            return str.split(self, sep)
        if len(self) == 0:
            return []
        return [
            [
                [x.replace("&slash;", "/") for x in token.split("/")]
                for token in sentence.split(" ")
            ]
            for sentence in str.split(self, "\n")
        ]


class Spelling(lazydict):
    ALPHA = "abcdefghijklmnopqrstuvwxyz"

    def __init__(self, path=""):
        self._path = path

    def load(self):
        for x in _read(self._path):
            x = x.split()
            dict.__setitem__(self, x[0], int(x[1]))

    @property
    def path(self):
        return self._path

    @property
    def language(self):
        return self._language

    @classmethod
    def train(self, s, path="spelling.txt"):
        model = {}
        for w in re.findall("[a-z]+", s.lower()):
            model[w] = w in model and model[w] + 1 or 1
        model = (f"{k} {v}" for k, v in sorted(model.items()))
        model = "\n".join(model)
        f = open(path, "w")
        f.write(model)
        f.close()

    def _edit1(self, w):
        split = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        delete, transpose, replace, insert = (
            [a + b[1:] for a, b in split if b],
            [a + b[1] + b[0] + b[2:] for a, b in split if len(b) > 1],
            [a + c + b[1:] for a, b in split for c in Spelling.ALPHA if b],
            [a + c + b[0:] for a, b in split for c in Spelling.ALPHA],
        )
        return set(delete + transpose + replace + insert)

    def _edit2(self, w):
        return set(e2 for e1 in self._edit1(w) for e2 in self._edit1(e1) if e2 in self)

    def _known(self, words=None):
        if words is None:
            words = []
        return set(w for w in words if w in self)

    def suggest(self, w):
        if len(self) == 0:
            self.load()
        if len(w) == 1:
            return [(w, 1.0)]  # I
        if w in PUNCTUATION:
            return [(w, 1.0)]  # .?!
        if w in string.whitespace:
            return [(w, 1.0)]  # \n
        if w.replace(".", "").isdigit():
            return [(w, 1.0)]  # 1.5
        candidates = (
            self._known([w])
            or self._known(self._edit1(w))
            or self._known(self._edit2(w))
            or [w]
        )
        candidates = [(self.get(c, 0.0), c) for c in candidates]
        s = float(sum(p for p, word in candidates) or 1)
        candidates = sorted(((p / s, word) for p, word in candidates), reverse=True)
        if w.istitle():  # Preserve capitalization
            candidates = [(word.title(), p) for p, word in candidates]
        else:
            candidates = [(word, p) for p, word in candidates]
        return candidates
