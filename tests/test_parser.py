"""
Unit tests for parser routing logic.

These tests use mocking to verify routing decisions without loading any models.

Run with: python -m pytest tests/test_parser.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch
from src.parser.parser_router import ParserRouter, ParserBackend, DOMAIN_PARSER_MAP


class TestDomainRouting:
    """Test that each domain routes to the correct backend."""

    def setup_method(self):
        self.router = ParserRouter(configs={})

    def test_maps_routes_to_paddleocr(self):
        assert self.router.get_backend_for_domain("maps") == ParserBackend.PADDLEOCR_VL

    def test_comics_routes_to_paddleocr(self):
        assert self.router.get_backend_for_domain("comics") == ParserBackend.PADDLEOCR_VL

    def test_engineering_drawing_routes_to_paddleocr(self):
        assert self.router.get_backend_for_domain("engineering_drawing") == ParserBackend.PADDLEOCR_VL

    def test_science_poster_routes_to_paddleocr(self):
        assert self.router.get_backend_for_domain("science_poster") == ParserBackend.PADDLEOCR_VL

    def test_infographics_routes_to_paddleocr(self):
        assert self.router.get_backend_for_domain("infographics") == ParserBackend.PADDLEOCR_VL

    def test_science_paper_routes_to_docling(self):
        assert self.router.get_backend_for_domain("science_paper") == ParserBackend.DOCLING

    def test_business_report_routes_to_docling(self):
        assert self.router.get_backend_for_domain("business_report") == ParserBackend.DOCLING

    def test_slide_routes_to_docling(self):
        assert self.router.get_backend_for_domain("slide") == ParserBackend.DOCLING

    def test_unknown_domain_defaults_to_paddleocr(self):
        assert self.router.get_backend_for_domain("unknown_domain") == ParserBackend.PADDLEOCR_VL

    def test_case_insensitive_routing(self):
        assert self.router.get_backend_for_domain("Maps") == ParserBackend.PADDLEOCR_VL
        assert self.router.get_backend_for_domain("MAPS") == ParserBackend.PADDLEOCR_VL

    def test_all_8_domains_covered(self):
        expected_domains = {
            "maps", "comics", "engineering_drawing", "infographics",
            "science_poster", "science_paper", "business_report", "slide"
        }
        for domain in expected_domains:
            # Should not raise KeyError or return None
            backend = self.router.get_backend_for_domain(domain)
            assert backend is not None


class TestRouterLazyInit:
    """Test that parsers are only initialized when needed."""

    def test_router_init_does_not_load_models(self):
        """Creating a router should not import or load any heavy models."""
        with patch("src.parser.paddleocr_vl.PaddleOCRVLParser") as mock_paddle:
            with patch("src.parser.docling_parser.DoclingParser") as mock_docling:
                router = ParserRouter(configs={})
                # Neither parser should have been instantiated
                mock_paddle.assert_not_called()
                mock_docling.assert_not_called()


class TestRouterOverride:
    """Test the backend override mechanism for ablation studies."""

    def test_override_changes_routing(self):
        router = ParserRouter(configs={})
        original = router.get_backend_for_domain("science_paper")
        assert original == ParserBackend.DOCLING

        router.override_backend("science_paper", ParserBackend.PADDLEOCR_VL)
        assert router.get_backend_for_domain("science_paper") == ParserBackend.PADDLEOCR_VL

        # Restore for other tests
        router.override_backend("science_paper", ParserBackend.DOCLING)
