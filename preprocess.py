import re
import emoji  # 1. Yahan import karna zaroori hai

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 2. Yeh line emojis ko text equivalents mein badal degi
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # URLs, mentions aur special characters ko hatana
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    
    text = text.strip()
    return text