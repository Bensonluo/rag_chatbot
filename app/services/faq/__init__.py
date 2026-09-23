"""FAQ fast path: semantic matching over curated question/answer pairs."""

from app.services.faq.store import FAQEntry, FAQService, create_faq_service, load_faq_entries

__all__ = ["FAQEntry", "FAQService", "create_faq_service", "load_faq_entries"]
