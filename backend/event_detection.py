
import re

# ── Cricket keyword set ────────────────────────────────────────────────────────

CRICKET_KEYWORDS = {
    "cricket", "batsman", "batter", "bowler", "fielder", "wicketkeeper",
    "wicket", "wickets", "stumps", "bail", "crease", "batting",
    "bowling", "fielding", "innings", "inning",
    "six", "sixes", "boundary", "boundaries",
    "century", "fifty", "half-century", "duck", "no-ball", "wide",
    "powerplay", "death overs", "free hit", "dot ball", "maiden",
    "caught", "bowled", "lbw", "stumped", "run out", "hit wicket",
    "dismissed", "appeal", "howzat", "review", "drs", "third umpire",
    "yorker", "bouncer", "googly", "spinner", "pacer", "all-rounder",
    "drive", "pull", "hook", "sweep", "cut shot", "cover drive",
    "slip", "gully", "mid-on", "mid-off", "fine leg", "square leg",
    "t20", "odi", "test match", "ipl", "world cup", "final", "semi-final",
    "scorecard", "target", "chase", "net run rate",
    "hat-trick", "partnership", "opening partnership",
    # Teams (high-confidence cricket context)
    "india", "australia", "pakistan", "england", "new zealand",
    "south africa", "west indies", "sri lanka", "bangladesh", "afghanistan",
    # Players (high-confidence)
    "kohli", "rohit", "dhoni", "bumrah", "shami", "jadeja", "siraj",
    "hardik", "pandya", "pant", "ashwin", "dube", "samson",
    "abhishek", "suryakumar", "axar", "arshdeep", "chakaravarthy",
    "ishan", "kishan", "gill", "tilak", "varma",
    "babar", "rizwan", "stokes", "root", "buttler", "archer",
    "warner", "smith", "starc", "cummins", "maxwell",
    "de kock", "rabada", "markram", "neesham", "santner",
    "allen", "ravindra", "phillips", "ferguson", "henry",
    # Venues & Tournaments
    "narendra modi", "wankhede", "eden gardens", "chepauk",
    "lords", "the oval", "mcg", "scg",
    "icc", "bcci", "champions",
}


AMBIGUOUS_CRICKET_WORDS = {
    "run", "runs", "over", "overs", "four", "pitch", "bat", "ball",
    "field", "score", "cover", "drive", "play", "match",
}

NON_CRICKET_SPORTS = {
    "football", "soccer", "goal", "penalty", "offside", "goalkeeper",
    "basketball", "hoop", "dunk", "three-pointer", "nba",
    "tennis", "serve", "forehand", "backhand", "ace", "wimbledon",
    "hockey", "puck", "nhl",
    "golf", "birdie", "eagle", "fairway", "putting",
    "rugby", "scrum", "touchdown",
    "baseball", "pitcher", "home run", "strikeout",
    "boxing", "knockout", "uppercut", "jab",
    "swimming", "freestyle", "breaststroke",
    "formula", "f1", "motorsport", "grand prix",
    "wrestling", "ufc", "mma",
    "volleyball", "badminton", "table tennis",
}


NON_ENGLISH_INDICATORS = [
    # Marathi/Hindi function words that appear in romanised transcripts
    r'\bna\b', r'\bka\b', r'\bki\b', r'\bko\b', r'\bse\b', r'\bye\b',
    r'\bho\b', r'\btha\b', r'\bkya\b', r'\bhai\b', r'\bkar\b',
    r'\baur\b', r'\bkuch\b', r'\bmain\b', r'\btum\b', r'\bapna\b',
    r'\bkaro\b', r'\bjao\b', r'\baana\b', r'\bjaana\b',
    # Romanised Marathi
    r'\bala\b', r'\bkela\b', r'\bghya\b', r'\bashe\b', r'\baahe\b',
    r'\btyala\b', r'\btyachi\b', r'\bamhi\b', r'\bapan\b',
]

KNOWN_PLAYERS = [
    "kohli", "rohit", "dhoni", "bumrah", "shami", "jadeja", "siraj",
    "hardik", "pandya", "pant", "ashwin", "dube", "samson",
    "abhishek sharma", "abhishek", "suryakumar", "axar", "arshdeep",
    "chakaravarthy", "ishan", "kishan", "gill", "shaw", "tilak", "varma",
    "babar", "rizwan", "stokes", "root", "buttler", "archer",
    "warner", "smith", "starc", "cummins", "maxwell", "hazlewood",
    "de kock", "rabada", "markram", "neesham", "santner",
    "allen", "ravindra", "phillips", "ferguson", "henry",
    "gambhir", "jay shah",
]

KNOWN_TEAMS = [
    "india", "pakistan", "australia", "england", "south africa",
    "new zealand", "west indies", "sri lanka", "bangladesh",
    "afghanistan", "zimbabwe", "ireland",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _word_set(text: str) -> set:
    """Tokenise text into a set of lowercase unigrams + bigrams."""
    words = re.findall(r"[a-z']+", text.lower())
    unigrams = set(words)
    bigrams  = {words[i] + " " + words[i + 1] for i in range(len(words) - 1)}
    return unigrams | bigrams


def _is_non_english(text: str) -> bool:
    if not text or len(text.split()) < 20:
        return False  # Too short to judge

    lower = text.lower()
    hit_count = 0
    for pat in NON_ENGLISH_INDICATORS:
        matches = len(re.findall(pat, lower))
        hit_count += matches
        if hit_count >= 6:  # 6+ non-English indicators = likely non-English
            return True

    # Also check ratio of recognised English words
    all_words = re.findall(r"[a-z]+", lower)
    if not all_words:
        return False

    # A very rough English word list (common function/content words)
    COMMON_ENGLISH = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "not", "no", "and", "but",
        "or", "if", "in", "on", "at", "to", "of", "for", "with", "as", "by",
        "from", "that", "this", "it", "he", "she", "they", "we", "you", "i",
        "his", "her", "their", "our", "its", "my", "your",
        "what", "when", "where", "who", "how", "which", "there", "here",
        "up", "out", "about", "into", "so", "than", "then", "just", "more",
        "also", "very", "get", "like", "go", "know", "see", "come", "say",
    }
    english_count = sum(1 for w in all_words if w in COMMON_ENGLISH)
    english_ratio = english_count / len(all_words)

    # If fewer than 8% of words are common English words, likely non-English
    if english_ratio < 0.08 and len(all_words) > 30:
        print(f"    [Lang detect] Low English ratio: {english_ratio:.2%} — likely non-English transcript")
        return True

    return False


def _count_cricket_hits(words: set) -> int:
    """Count STRONG cricket keyword hits (excluding ambiguous words)."""
    strong = CRICKET_KEYWORDS - AMBIGUOUS_CRICKET_WORDS
    return len(words & strong)


def _count_ambiguous_hits(words: set) -> int:
    """Count ambiguous words that need strong context to matter."""
    return len(words & AMBIGUOUS_CRICKET_WORDS)


# ── Main domain check functions ────────────────────────────────────────────────

def check_file_domain(text: str, filename: str, is_transcript: bool = False) -> dict:
    if not text or not text.strip():
        return {
            "is_cricket": False, "reject": False,
            "message": f"No text content found in '{filename}'.",
        }

    words = _word_set(text)
    cs = _count_cricket_hits(words)      # strong cricket keywords
    ca = _count_ambiguous_hits(words)    # ambiguous words
    ns = len(words & NON_CRICKET_SPORTS)

    total_words = len(text.split())

    print(f"    [Domain '{filename}'] strong_cricket={cs}, ambiguous={ca}, "
          f"non_cricket_sports={ns}, words={total_words}, is_transcript={is_transcript}")

    # Step 1: If it heavily mentions another sport, reject it.
    if ns >= 3 and cs == 0:
        return {
            "is_cricket": False, "reject": True,
            "message": f"'{filename}' appears to be about a different sport. File omitted.",
        }

    # Step 2: RELAXED RULE - If it has ANY cricket words, or even just ambiguous words, accept it.
    # The LLM is smart enough to figure it out later. Let the data through.
    if cs >= 1 or ca >= 1 or total_words < 50:
        return {
            "is_cricket": True, "reject": False,
            "message": f"'{filename}' — Content accepted.",
        }

    # Step 3: Default accept for safety. We'd rather pass garbage to the LLM than drop good files.
    return {
        "is_cricket": True, "reject": False,
        "message": f"'{filename}' — Content accepted.",
    }
def check_image_caption_domain(raw_caption: str) -> bool:
    # Import here to avoid circular import
    from captioning import is_cricket_visual
    return is_cricket_visual(raw_caption)


def run_domain_detection(text: str) -> dict:
    """Single-text combined domain check (legacy / fallback use)."""
    words = _word_set(text)
    cs = _count_cricket_hits(words)
    ns = len(words & NON_CRICKET_SPORTS)
    if ns >= 2 and cs <= 1:
        return {
            "domain": "NON_CRICKET", "supported": False,
            "message": "Content appears to be from a different sport.",
        }
    if cs >= 2:
        conf = "HIGH" if cs >= 8 else ("MEDIUM" if cs >= 4 else "LOW")
        return {
            "domain": "CRICKET", "supported": True,
            "message": f"Cricket detected ({conf} confidence).",
        }
    return {
        "domain": "UNKNOWN", "supported": False,
        "message": "Domain unclear — no cricket content detected.",
    }


def get_ner_entities_simple(text: str) -> dict:
    lower = text.lower()
    players = list({
        p.title() for p in KNOWN_PLAYERS
        if re.search(r'\b' + re.escape(p) + r'\b', lower)
    })
    teams = list({
        t.title() for t in KNOWN_TEAMS
        if re.search(r'\b' + re.escape(t) + r'\b', lower)
    })
    return {"players": players, "teams": teams}