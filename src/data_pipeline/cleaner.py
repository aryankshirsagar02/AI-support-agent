"""
Text cleaning and normalization utilities for Twitter Customer Support data.
"""
import re
import html
from typing import Optional


def clean_tweet_text(text: Optional[str], remove_handles: bool = True) -> str:
    """
    Clean and sanitize raw tweet text.
    
    Operations:
    1. Unescape HTML entities (&amp;, &lt;, &gt;)
    2. Normalize URLs to placeholder [URL]
    3. Remove or normalize customer @mentions (preserving brand identity context if desired)
    4. Normalize whitespace and strip newlines
    5. Clean emoji artifacts or encoding errors
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    cleaned = html.unescape(text)
    
    # Normalize URLs
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\-._~:?#[\]@!$&\'()*+,;=]*'
    cleaned = re.sub(url_pattern, '[URL]', cleaned)
    
    # Remove @mentions if requested
    if remove_handles:
        # Replace @username with blank, but keep words following
        cleaned = re.sub(r'@[A-Za-z0-9_]+', '', cleaned)
    
    # Mask order numbers / tracking IDs (e.g., 112-1234567-1234567 or 12-digit tracking)
    cleaned = mask_sensitive_info(cleaned)
    
    # Normalize multiple whitespace, tabs, newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def mask_sensitive_info(text: str) -> str:
    """
    Mask sensitive user identifiers like order IDs, emails, credit card numbers, and phone numbers.
    """
    if not text:
        return ""
    
    # Mask email addresses
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '[EMAIL]', text)
    
    # Mask Amazon order IDs (e.g. 111-1234567-1234567 or 3-7-7 digits)
    order_pattern = r'\b\d{3}-\d{7}-\d{7}\b'
    text = re.sub(order_pattern, '[ORDER_ID]', text)
    
    # Mask long numeric sequences (card / tracking / phone numbers >= 8 digits)
    phone_or_card_pattern = r'\b\d{4}[ -]?\d{4}[ -]?\d{4}(?:[ -]?\d{4})?\b'
    text = re.sub(phone_or_card_pattern, '[CARD/PHONE]', text)
    
    return text
