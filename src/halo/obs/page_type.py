"""Page type classification for HALO Agent.

Simple heuristic classifier for page types to support caching/macros.
"""

from typing import Optional
from urllib.parse import urlparse


# Page type constants
PAGE_TYPE_HOME = 'home'
PAGE_TYPE_SEARCH = 'search'
PAGE_TYPE_RESULTS = 'results'
PAGE_TYPE_PRODUCT = 'product'
PAGE_TYPE_CART = 'cart'
PAGE_TYPE_CHECKOUT = 'checkout'
PAGE_TYPE_EMAIL_LIST = 'email_list'
PAGE_TYPE_EMAIL_COMPOSE = 'email_compose'
PAGE_TYPE_EMAIL_VIEW = 'email_view'
PAGE_TYPE_CALENDAR = 'calendar'
PAGE_TYPE_FORM = 'form'
PAGE_TYPE_LOGIN = 'login'
PAGE_TYPE_PROFILE = 'profile'
PAGE_TYPE_SETTINGS = 'settings'
PAGE_TYPE_UNKNOWN = 'unknown'


def classify_page_type(url: str, title: str = '') -> str:
    """Classify page type from URL and title.

    Args:
        url: Current page URL
        title: Page title (optional)

    Returns:
        Page type string constant
    """
    url_lower = url.lower()
    title_lower = title.lower() if title else ''
    parsed = urlparse(url_lower)
    path = parsed.path
    query = parsed.query

    # Login pages
    if any(x in path for x in ['/login', '/signin', '/sign-in', '/auth']):
        return PAGE_TYPE_LOGIN
    if any(x in title_lower for x in ['login', 'sign in', 'log in']):
        return PAGE_TYPE_LOGIN

    # Checkout pages (high-stakes)
    if any(x in path for x in ['/checkout', '/payment', '/order', '/purchase']):
        return PAGE_TYPE_CHECKOUT
    if any(x in title_lower for x in ['checkout', 'payment', 'order confirmation']):
        return PAGE_TYPE_CHECKOUT

    # Cart pages
    if any(x in path for x in ['/cart', '/basket', '/bag']):
        return PAGE_TYPE_CART
    if any(x in title_lower for x in ['cart', 'basket', 'shopping bag']):
        return PAGE_TYPE_CART

    # Product pages
    if any(x in path for x in ['/product', '/item', '/dp/', '/p/']):
        return PAGE_TYPE_PRODUCT
    if any(x in title_lower for x in ['product details', 'buy now']):
        return PAGE_TYPE_PRODUCT

    # Search / Results
    if 'search' in query or 'q=' in query or 's=' in query:
        return PAGE_TYPE_SEARCH
    if any(x in path for x in ['/search', '/results', '/browse']):
        return PAGE_TYPE_RESULTS

    # Email pages
    if any(x in url_lower for x in ['mail', 'inbox', 'email']):
        if 'compose' in path or 'new' in path or 'write' in path:
            return PAGE_TYPE_EMAIL_COMPOSE
        if any(x in path for x in ['/view', '/read', '/message/']):
            return PAGE_TYPE_EMAIL_VIEW
        return PAGE_TYPE_EMAIL_LIST

    # Calendar pages
    if any(x in url_lower for x in ['calendar', 'schedule', 'event']):
        return PAGE_TYPE_CALENDAR

    # Form pages
    if any(x in path for x in ['/contact', '/form', '/submit', '/register', '/signup']):
        return PAGE_TYPE_FORM

    # Profile/Settings
    if any(x in path for x in ['/profile', '/account', '/user']):
        return PAGE_TYPE_PROFILE
    if any(x in path for x in ['/settings', '/preferences', '/config']):
        return PAGE_TYPE_SETTINGS

    # Home page (root path)
    if path in ['', '/', '/home', '/index']:
        return PAGE_TYPE_HOME

    return PAGE_TYPE_UNKNOWN


def is_high_stakes_page(page_type: str) -> bool:
    """Check if page type is high-stakes (requires manager)."""
    return page_type in {PAGE_TYPE_CHECKOUT, PAGE_TYPE_CART, PAGE_TYPE_LOGIN}


def is_form_page(page_type: str) -> bool:
    """Check if page type involves form filling."""
    return page_type in {
        PAGE_TYPE_FORM, PAGE_TYPE_EMAIL_COMPOSE,
        PAGE_TYPE_LOGIN, PAGE_TYPE_CHECKOUT
    }


# Confirmation signals that indicate task completion
CONFIRMATION_SIGNALS = [
    # Order/purchase confirmations
    'order confirmed', 'order placed', 'order complete', 'purchase complete',
    'thank you for your order', 'order successful', 'payment successful',
    'transaction complete', 'booking confirmed', 'reservation confirmed',
    # Email confirmations
    'email sent', 'message sent', 'mail sent', 'sent successfully',
    'your message has been sent', 'email delivered',
    # Save/update confirmations
    'saved', 'saved successfully', 'changes saved', 'profile updated',
    'settings saved', 'preferences updated', 'update successful',
    # Form submissions
    'submitted', 'form submitted', 'submission successful',
    'thank you for submitting', 'request submitted',
    # Calendar/event confirmations
    'event created', 'event saved', 'invitation sent', 'meeting scheduled',
    'appointment confirmed', 'added to calendar',
    # Account actions
    'account created', 'registration complete', 'signed up successfully',
    'password changed', 'password updated',
    # General success
    'success', 'completed', 'done', 'confirmed',
]


def has_confirmation_signal(axtree_text: str, title: str = '') -> bool:
    """Check if the page shows a confirmation signal indicating task completion.
    
    Args:
        axtree_text: Flattened AXTree text or page content
        title: Page title
        
    Returns:
        True if a confirmation signal is detected
    """
    text_lower = (axtree_text + ' ' + title).lower()
    
    for signal in CONFIRMATION_SIGNALS:
        if signal in text_lower:
            return True
    
    return False


def extract_confirmation_text(axtree_text: str) -> str:
    """Extract the confirmation message if present.
    
    Args:
        axtree_text: Flattened AXTree text
        
    Returns:
        Confirmation message or empty string
    """
    text_lower = axtree_text.lower()
    
    for signal in CONFIRMATION_SIGNALS:
        if signal in text_lower:
            # Find the sentence containing the signal
            idx = text_lower.find(signal)
            start = max(0, idx - 50)
            end = min(len(axtree_text), idx + len(signal) + 50)
            return axtree_text[start:end].strip()
    
    return ""
