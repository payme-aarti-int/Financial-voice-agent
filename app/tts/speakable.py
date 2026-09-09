from __future__ import annotations
 
import re
 
MONTHS = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}
 
 
def _spoken_magnitude(value: float) -> str:
   
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} billion".replace(".00", "")
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f} million".replace(".00", "")
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f} thousand".replace(".0", "")
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"
 
 
def spoken_money(value: float, currency: str = "dollars") -> str:
    return f"{_spoken_magnitude(value)} {currency}"
 
 
def _currency_to_words(match: re.Match) -> str:
    raw = match.group(1).replace(",", "")
    try:
        return spoken_money(float(raw))
    except ValueError:
        return match.group(0)
 
 
def _bare_number_to_words(match: re.Match) -> str:
    raw = match.group(0).replace(",", "")
    try:
        return _spoken_magnitude(float(raw))
    except ValueError:
        return match.group(0)
 
 
def _iso_month_to_words(match: re.Match) -> str:
    year, month = match.group(1), match.group(2)
    return f"{MONTHS.get(month, month)} {year}"
 
 
def to_speakable(text: str) -> str:
    if not text:
        return ""
 
    result = text
 
    result = re.sub(r"\*\*(.+?)\*\*", r"\1", result)
    result = re.sub(r"\*(.+?)\*", r"\1", result)
    result = re.sub(r"`(.+?)`", r"\1", result)
    result = re.sub(r"^#{1,6}\s*", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*[-*+]\s+", "", result, flags=re.MULTILINE)
    result = re.sub(r"^\s*\d+\.\s+", "", result, flags=re.MULTILINE)
 
    result = re.sub(r"\b(\d{4})-(\d{2})-\d{2}\b", _iso_month_to_words, result)
    result = re.sub(r"\b(\d{4})-(\d{2})\b", _iso_month_to_words, result)
 
    result = re.sub(r"\bQ([1-4])\b", r"quarter \1", result)
 
    result = re.sub(
        r"\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)",
        _currency_to_words,
        result,
    )
 
    result = re.sub(r"([\d.]+)\s?%", r"\1 percent", result)
 
    result = re.sub(
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b", _bare_number_to_words, result
    )
    result = re.sub(r"\b\d{5,}(?:\.\d+)?\b", _bare_number_to_words, result)
    result = re.sub(
        r"\b(?!19\d{2}\b|20\d{2}\b)\d{4}(?:\.\d+)?\b",
        _bare_number_to_words,
        result,
    )
 
    result = re.sub(r"\n+", " ", result)
    result = re.sub(r"\s{2,}", " ", result)
 
    return result.strip()
 
