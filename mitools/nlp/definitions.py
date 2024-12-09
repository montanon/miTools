import re
import string

from nltk.corpus import wordnet

VERB, NOUN, ADJ, ADV = wordnet.VERB, wordnet.NOUN, wordnet.ADJ, wordnet.ADV
PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")
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
REPLACEMENTS = {
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
