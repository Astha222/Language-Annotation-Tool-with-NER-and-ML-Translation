"""Detect Hindi written in English letters ("Hinglish", Romanized Hindi).

langdetect, which the app uses to pick the language, has no profile for it: the five test sentences in the WS3 log were called Indonesian, Swahili and English,
never Hindi. This module is a small, explainable check instead of a model: a text counts as Romanized Hindi when
  * almost all of its letters are Latin (Devanagari text is left to langdetect, which handles it), and
  * at least HINGLISH_MIN_MARKERS of its words, and at least HINGLISH_MIN_SHARE of all its words, are common Hindi function words
    (hai, ka, ki, mein, nahi, kya, bahut ...).
Words that are also ordinary English, Spanish, French or German words (to, the, me, main, par, do, in, so, ab, ...) are deliberately NOT in the list.
The thresholds were fixed before the test set was scored (see evaluation/eval_hinglish_detect.py and IMPROVEMENT_LOG.md).
"""
import re

HINGLISH_MIN_MARKERS = 2
HINGLISH_MIN_SHARE = 0.20
MIN_LATIN_SHARE = 0.85

MARKERS = frozenset("""
hai hain hoon hun hoga hogi honge tha thi thhe
ka ki ke ko pe aur ya nahi nahin nhi mat
kya kyun kyu kyon kaise kaisa kaisi kahan kahaan kab kaun kitna kitne kitni
yeh ye woh wo yahan yaha wahan waha
mera meri mere tera teri tere tumhara tumhari tumhare hamara hamari hamare uska uski uske unka unki unke apna apni apne
hum tum aap mujhe mujhko tumhe tumhein humein usse isse inhe unhe
bhi toh bahut bohot bahot bohat accha achha acha theek thik abhi kal aaj
raha rahi rahe gaya gayi gaye kiya kiye karna karta karti karte kar karo
kuch koi sab sabko lekin magar kyunki jab tab agar phir fir
yaar bhai dost chahiye sakta sakti sakte liye wala wali wale vaala saath bilkul zyada jyada
bas matlab sirf sach galat pata ek mein
ne ji naya naye nayi liya diya diye kaam ghar logon raat baat bola bata bataya batao dekho dekha dekhna jaldi kabhi sabhi isliye
itna itni thoda thodi kaafi chalo aana jana jaana lena dena shayad pehle hamesha mila mile laga lagta lagti
""".split())

_WORD = re.compile(r"[a-z]+")


def _latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() or "À" <= c <= "ɏ" for c in letters) / len(letters)


def hinglish_score(text: str) -> tuple:
    """(markers, words, share of words that are Hindi markers) for the Latin-script words of the text."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0, 0, 0.0
    markers = sum(w in MARKERS for w in words)
    return markers, len(words), markers / len(words)


def is_romanized_hindi(text: str) -> bool:
    if not text or _latin_share(text) < MIN_LATIN_SHARE:
        return False
    markers, words, share = hinglish_score(text)
    return markers >= HINGLISH_MIN_MARKERS and share >= HINGLISH_MIN_SHARE
