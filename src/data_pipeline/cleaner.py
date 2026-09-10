"""
Text cleaning, normalization, and sensitive entity masking utilities for SupportIQ AI.
"""
import re
import html
from typing import Optional, Dict, Tuple


def mask_sensitive_info(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Mask sensitive user identifiers like order IDs, emails, credit card numbers,
    account numbers, tracking IDs, and phone numbers.
    Returns the sanitized text along with counts of masked entities.
    """
    if not text:
        return "", {}
    
    counts: Dict[str, int] = {
        "email": 0,
        "phone": 0,
        "order_id": 0,
        "account_number": 0,
        "card_number": 0,
        "url": 0,
    }

    # 1. Mask URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-._~:?#[\]@!$&\'()*+,;=]*'
    urls_found = len(re.findall(url_pattern, text))
    if urls_found:
        counts["url"] = urls_found
        text = re.sub(url_pattern, '[URL]', text)

    # 2. Mask email addresses
    email_pattern = r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b'
    emails_found = len(re.findall(email_pattern, text))
    if emails_found:
        counts["email"] = emails_found
        text = re.sub(email_pattern, '[EMAIL]', text)

    # 3. Mask Order IDs (e.g., Amazon 112-1234567-1234567 or generic #12345678)
    order_pattern = r'\b\d{3}-\d{7}-\d{7}\b'
    orders_found = len(re.findall(order_pattern, text))
    if orders_found:
        counts["order_id"] = orders_found
        text = re.sub(order_pattern, '[ORDER_ID]', text)

    # 4. Mask Credit Card Numbers (13-16 digits with optional spaces or dashes)
    card_pattern = r'\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{15,16}\b'
    cards_found = len(re.findall(card_pattern, text))
    if cards_found:
        counts["card_number"] = cards_found
        text = re.sub(card_pattern, '[CARD_NUMBER]', text)

    # 5. Mask Phone numbers (US/International standard formats)
    phone_pattern = r'\b(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b'
    phones_found = len(re.findall(phone_pattern, text))
    if phones_found:
        counts["phone"] = phones_found
        text = re.sub(phone_pattern, '[PHONE]', text)

    # 6. Mask 8-12 digit Account / Reference / Tracking numbers
    account_pattern = r'\b(?:ACC|ACCT|REF|TRACK)?[ -]?#?\d{8,12}\b'
    accounts_found = len(re.findall(account_pattern, text))
    if accounts_found:
        counts["account_number"] = accounts_found
        text = re.sub(account_pattern, '[ACCOUNT_NUMBER]', text)

    return text, counts


def clean_tweet_text(text: Optional[str], remove_handles: bool = True) -> str:
    """
    Clean and sanitize raw tweet text.
    
    Operations:
    1. Unescape HTML entities (&amp;, &lt;, &gt;)
    2. Remove or normalize customer @mentions
    3. Mask sensitive PII entities (email, phone, order ID, cards)
    4. Normalize multiple whitespace and strip newlines
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    cleaned = html.unescape(text)
    
    # Remove @mentions if requested (e.g. @AmazonHelp, @115712)
    if remove_handles:
        cleaned = re.sub(r'@[A-Za-z0-9_]+', '', cleaned)
    
    # Mask sensitive entities
    cleaned, _ = mask_sensitive_info(cleaned)
    
    # Normalize whitespace, tabs, and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned
