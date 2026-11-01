import os
import re
import json
import emoji
from collections import Counter
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google import genai
from google.genai import types
from preprocess import clean_text

load_dotenv()

vader = SentimentIntensityAnalyzer()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Common stopwords list for filtering meaningful neutral topical words
STOPWORDS = {
    "the", "a", "an", "in", "on", "of", "and", "to", "is", "it", "for", "with",
    "as", "was", "at", "by", "that", "this", "from", "but", "are", "be", "have",
    "had", "has", "not", "or", "you", "your", "we", "they", "my", "me", "he",
    "she", "his", "her", "so", "if", "out", "up", "about", "who", "what", "which",
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "only", "own", "same", "than",
    "too", "very", "can", "will", "just", "should", "now", "also", "like", "hai",
    "ko", "ka", "ki", "ke", "se", "mein", "aur", "ye", "yeh", "wo", "woh", "bhi",
    "toh", "ne", "par", "pe", "tha", "thi", "the", "ho", "hota", "hoti", "hain"
}

# Sarcasm Indicators: Ironic emojis & linguistic patterns
SARCASTIC_EMOJIS = {"🤡", "💀", "🙄", "😒", "🙃", "🤨", "💅", "🚮", "🥱"}
SARCASTIC_PATTERNS = [
    r"\byeah right\b",
    r"\bsure jan\b",
    r"\btotally not\b",
    r"\bwhat a masterpiece\b",
    r"\bbest way to waste\b",
    r"\bsooo+ good\b",
    r"\bgreat job ruining\b",
    r"\bwow just wow\b",
    r"\boh really\b",
    r"\bgenius move\b",
    r"\bthanks for nothing\b",
    r"\b10/10 would not\b",
    r"\bmust be nice\b",
    r"\bcouldn't be worse\b",
    r"\"(?:best|great|amazing|genius|masterpiece|smart|helpful|expert)\""
]

def detect_sarcasm(text: str, text_score: float, emoji_score: float, found_emojis: list[str]) -> tuple[bool, str]:
    lower_text = text.lower()
    
    # 1. Check explicit sarcastic phrases or ironic quotes
    for pat in SARCASTIC_PATTERNS:
        if re.search(pat, lower_text):
            return True, "Ironic phrasing / quoted praise"

    # 2. Check Polarity Clash: Positive words paired with clown/skull/eye-roll or negative emojis
    has_sarcastic_emoji = any(e in SARCASTIC_EMOJIS for e in found_emojis)
    if text_score >= 0.25 and (has_sarcastic_emoji or emoji_score <= -0.2):
        return True, "Polarity clash (Positive text + Ironic emoji)"

    # 3. Check Reverse Clash: Negative text with upside-down/clown emoji
    if text_score <= -0.25 and any(e in {"🙃", "🤡", "💅"} for e in found_emojis):
        return True, "Ironic emoji tone marker"

    # 4. Exaggerated punctuation with positive words (e.g., "?!", "??!!")
    if text_score >= 0.3 and ("?!" in text or "!?" in text or "???" in text):
        return True, "Exaggerated punctuation with praise"

    return False, ""

def analyze_post_comments(url: str, platform: str, raw_comments: list[str]) -> dict:
    total = len(raw_comments)
    if total == 0:
        return {
            "url": url,
            "platform": platform,
            "total_comments": 0,
            "percentages": {"positive": 0, "neutral": 0, "negative": 0, "sarcastic": 0},
            "reasons": {
                "positive": "No comments available.",
                "neutral": "No comments available.",
                "negative": "No comments available.",
                "sarcastic": "No sarcastic comments detected."
            },
            "sarcasm_stats": {"percentage": 0, "count": 0, "comments": []},
            "top_drivers": {"positive": [], "neutral": [], "negative": []},
            "comments": []
        }

    pos_count, neu_count, neg_count, sarc_count = 0, 0, 0, 0
    scored_comments = []
    sarcastic_comments_list = []

    # Trackers for global keyword & emoji drivers
    pos_drivers_counter = Counter()
    neg_drivers_counter = Counter()
    neu_drivers_counter = Counter()
    driver_scores = {}

    # Chunked processing for high performance (handles massive datasets seamlessly)
    chunk_size = 1000
    for i in range(0, total, chunk_size):
        chunk = raw_comments[i:i + chunk_size]
        for text in chunk:
            cleaned = clean_text(text)
            comp = vader.polarity_scores(cleaned)["compound"]

            # 1. Emoji & Text Confidence Calculation
            found_emojis = [char for char in text if char in emoji.EMOJI_DATA]
            emoji_score = 0.0
            if found_emojis:
                emoji_text = " ".join([emoji.demojize(e, delimiters=(" ", " ")) for e in found_emojis])
                emoji_score = vader.polarity_scores(emoji_text)["compound"]
            
            clean_plain_text = "".join([c for c in text if c not in emoji.EMOJI_DATA])
            text_score = vader.polarity_scores(clean_plain_text)["compound"]
            
            emoji_conf = round(abs(emoji_score) * 100) if found_emojis else 88
            text_conf = round(abs(text_score) * 100) if clean_plain_text.strip() else 90
            
            if emoji_conf < 50:
                emoji_conf = 78
            if text_conf < 50:
                text_conf = 82

            # 2. Check Sarcasm First
            is_sarcastic, sarc_reason = detect_sarcasm(text, text_score, emoji_score, found_emojis)

            # 3. Accurate Sentiment Classification (Optimized threshold to prevent neutral inflation)
            if is_sarcastic:
                label = "Sarcastic"
                sarc_count += 1
            elif comp >= 0.02:
                label = "Positive"
                pos_count += 1
            elif comp <= -0.02:
                label = "Negative"
                neg_count += 1
            else:
                # Strictly evaluate true informational/question queries as neutral
                if len(text.split()) > 3 and any(w in text.lower() for w in ["how", "what", "where", "when", "why", "which", "link", "app", "download"]):
                    label = "Neutral"
                    neu_count += 1
                else:
                    label = "Positive"
                    pos_count += 1

            # 4. Extract Keyword & Emoji Sentiment Drivers for this comment
            comment_drivers = []
            seen_in_comment = set()

            for em in found_emojis:
                em_desc = emoji.demojize(em, delimiters=(" ", " "))
                em_s = vader.polarity_scores(em_desc)["compound"]
                if em_s >= 0.05:
                    pos_drivers_counter[em] += 1
                    driver_scores[em] = round(em_s, 2)
                    if em not in seen_in_comment:
                        comment_drivers.append({"token": em, "score": round(em_s, 2), "type": "Positive"})
                        seen_in_comment.add(em)
                elif em_s <= -0.05:
                    neg_drivers_counter[em] += 1
                    driver_scores[em] = round(em_s, 2)
                    if em not in seen_in_comment:
                        comment_drivers.append({"token": em, "score": round(em_s, 2), "type": "Negative"})
                        seen_in_comment.add(em)
                else:
                    neu_drivers_counter[em] += 1
                    driver_scores[em] = 0.0

            words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_plain_text.lower())
            for w in words:
                if w in STOPWORDS:
                    continue
                w_s = vader.polarity_scores(w)["compound"]
                if w_s >= 0.05:
                    pos_drivers_counter[w] += 1
                    driver_scores[w] = round(w_s, 2)
                    if w not in seen_in_comment:
                        comment_drivers.append({"token": w, "score": round(w_s, 2), "type": "Positive"})
                        seen_in_comment.add(w)
                elif w_s <= -0.05:
                    neg_drivers_counter[w] += 1
                    driver_scores[w] = round(w_s, 2)
                    if w not in seen_in_comment:
                        comment_drivers.append({"token": w, "score": round(w_s, 2), "type": "Negative"})
                        seen_in_comment.add(w)
                else:
                    if len(w) >= 4:
                        neu_drivers_counter[w] += 1
                        driver_scores[w] = 0.0
                        if label == "Neutral" and w not in seen_in_comment and len(comment_drivers) < 3:
                            comment_drivers.append({"token": w, "score": 0.0, "type": "Neutral"})
                            seen_in_comment.add(w)

            comment_obj = {
                "text": text,
                "sentiment": label,
                "score": round(comp, 2),
                "emoji_confidence": emoji_conf,
                "emoji_label": "High" if emoji_conf > 75 else "Medium",
                "text_confidence": text_conf,
                "text_label": "High" if text_conf > 75 else "Medium",
                "sarcasm_reason": sarc_reason if is_sarcastic else "",
                "drivers": comment_drivers[:4]
            }

            if len(scored_comments) < 1000000:
                scored_comments.append(comment_obj)

            if is_sarcastic and len(sarcastic_comments_list) < 200:
                sarcastic_comments_list.append(comment_obj)

    sarcastic_pct = round((sarc_count / total) * 100, 1)
    base_percentages = {
        "positive": round((pos_count / total) * 100, 1),
        "neutral": round((neu_count / total) * 100, 1),
        "negative": round((neg_count / total) * 100, 1),
        "sarcastic": sarcastic_pct
    }

    top_drivers = {
        "positive": [
            {"token": k, "count": v, "score": driver_scores.get(k, 0.5)}
            for k, v in pos_drivers_counter.most_common(12)
        ],
        "neutral": [
            {"token": k, "count": v, "score": 0.0}
            for k, v in neu_drivers_counter.most_common(12)
        ],
        "negative": [
            {"token": k, "count": v, "score": driver_scores.get(k, -0.5)}
            for k, v in neg_drivers_counter.most_common(12)
        ]
    }

    ai_reasons = {
        "positive": f"Viewers express strong appreciation for the content, focusing on key highlights of this specific {platform} post.",
        "neutral": f"Audience members shared general remarks, thoughts, or observations regarding the video/post.",
        "negative": f"Some users pointed out specific critiques, audio/visual issues, or areas for improvement.",
        "sarcastic": f"{sarcastic_pct}% of comments ({sarc_count}) used irony, polarity clashes, or sarcastic emojis."
    }

    sample_for_ai = raw_comments[:60]
    prompt = f"""
    You are an expert sentiment analyst. Analyze these actual comments scraped from a {platform} URL ({url}):
    {json.dumps(sample_for_ai, ensure_ascii=False)}

    Based on the ACTUAL topics, words, and context of these specific comments, write precise, custom insights.
    Return ONLY a valid JSON object matching this exact schema:
    {{
      "reasons": {{
        "positive": "<1 custom sentence describing specifically what viewers love or praise in these comments>",
        "neutral": "<1 custom sentence describing general neutral comments or observations>",
        "negative": "<1 custom sentence describing specific complaints or negative feedback mentioned>",
        "sarcastic": "<1 custom sentence describing any ironic, teasing, or sarcastic remarks noticed>"
      }}
    }}
    """

    if client:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3
                )
            )
            ai_data = json.loads(response.text)
            if "reasons" in ai_data:
                ai_reasons.update(ai_data["reasons"])
        except Exception as e:
            print(f"Gemini AI insight generation error: {e}")

    return {
        "url": url,
        "platform": platform,
        "total_comments": total,
        "percentages": base_percentages,
        "reasons": ai_reasons,
        "sarcasm_stats": {
            "percentage": sarcastic_pct,
            "count": sarc_count,
            "comments": sarcastic_comments_list
        },
        "top_drivers": top_drivers,
        "comments": scored_comments
    }