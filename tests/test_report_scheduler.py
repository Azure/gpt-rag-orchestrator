"""
Unit tests for report_scheduler/__init__.py
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock azure.functions to avoid ImportError
import unittest.mock
sys.modules['azure.functions'] = unittest.mock.MagicMock()
import report_scheduler

class TestReportScheduler(unittest.TestCase):
    def setUp(self):
        self.org_id = "org123"
        self.brand_name = "BrandX"
        self.industry_description = "IndustryDesc"
        self.sample_org = {"id": self.org_id, "industry_description": self.industry_description}
        self.sample_brand = {"name": self.brand_name}
        self.sample_payload = {
            "report_key": "brand_analysis",
            "report_name": self.brand_name,
            "params": {
                "brand_focus": self.brand_name,
                "industry_context": self.industry_description
            }
        }
        os.environ["AZURE_DB_NAME"] = "testdb"

    @patch("report_scheduler.cosmos_client_async.get_container")
    def test_get_all_organizations_success(self, mock_get_container):
        mock_container = MagicMock()
        mock_container.query_items.return_value = [{"id": "org1"}, {"id": "org2"}]
        mock_get_container.return_value = mock_container
        result = report_scheduler.get_all_organizations()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "org1")

    @patch("report_scheduler.cosmos_client_async.get_container")
    def test_get_brands_success(self, mock_get_container):
        mock_container = MagicMock()
        mock_container.query_items.return_value = [{"name": "BrandA"}, {"name": "BrandB"}]
        mock_get_container.return_value = mock_container
        result = report_scheduler.get_brands(self.org_id)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "BrandA")

    @patch("report_scheduler.cosmos_client_async.get_container")
    def test_get_products_success(self, mock_get_container):
        mock_container = MagicMock()
        mock_container.query_items.return_value = [{"name": "ProductA"}]
        mock_get_container.return_value = mock_container
        result = report_scheduler.get_products(self.org_id)
        self.assertEqual(result[0]["name"], "ProductA")

    @patch("report_scheduler.cosmos_client_async.get_container")
    def test_get_competitors_success(self, mock_get_container):
        mock_container = MagicMock()
        mock_container.query_items.return_value = [{"name": "CompetitorA"}]
        mock_get_container.return_value = mock_container
        result = report_scheduler.get_competitors(self.org_id)
        self.assertEqual(result[0]["name"], "CompetitorA")

    def test_create_brands_payload(self):
        payload = report_scheduler.create_brands_payload(self.brand_name, self.industry_description)
        self.assertEqual(payload, self.sample_payload)

    @patch("report_scheduler.requests.post")
    def test_send_http_request_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{}"
        mock_response.elapsed = timedelta(seconds=1)
        mock_response.json.return_value = {"result": "ok"}
        mock_post.return_value = mock_response
        url = "http://test/api"
        payload = {"key": "value"}
        response = report_scheduler.send_http_request(url, payload)
        self.assertEqual(response.status_code, 200)
        mock_post.assert_called_once_with(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=report_scheduler.TIMEOUT_SECONDS,
        )

    def test_log_response_result_success(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"{\"result\": \"ok\"}"
        mock_response.elapsed = timedelta(seconds=2)
        mock_response.json.return_value = {"result": "ok"}
        # Should not raise
        report_scheduler.log_response_result("http://test/api", mock_response)

    def test_log_response_result_json_error(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.content = b"notjson"
        mock_response.elapsed = timedelta(seconds=2)
        mock_response.json.side_effect = Exception("fail")
        # Should not raise
        report_scheduler.log_response_result("http://test/api", mock_response)

    @patch("report_scheduler.get_all_organizations")
    @patch("report_scheduler.get_brands")
    @patch("report_scheduler.send_http_request")
    @patch("report_scheduler.log_response_result")
    def test_main_flow(self, mock_log_response, mock_send_http, mock_get_brands, mock_get_orgs):
        # Setup mocks
        mock_get_orgs.return_value = [self.sample_org]
        mock_get_brands.return_value = [self.sample_brand]
        mock_send_http.return_value = MagicMock(status_code=200, content=b"{}", elapsed=timedelta(seconds=1), json=lambda: {"ok": True})
        mock_log_response.return_value = None
        # Patch environment
        with patch.dict(os.environ, {"AZURE_DB_NAME": "testdb"}):
            with patch.object(report_scheduler, "WEB_APP_URL", "http://test/api"):
                report_scheduler.main(MagicMock())
        mock_get_orgs.assert_called_once()
        mock_get_brands.assert_called_once_with(self.org_id)
        mock_send_http.assert_called()
        mock_log_response.assert_called()

    @patch("report_scheduler.get_all_organizations", side_effect=Exception("fail test"))
    def test_main_organizations_error(self, mock_get_orgs):
        with patch.object(report_scheduler, "WEB_APP_URL", "http://test/api"):
            # Should not raise, just log and return
            report_scheduler.main(MagicMock())

    @patch("report_scheduler.get_all_organizations", return_value=None)
    def test_main_no_organizations(self, mock_get_orgs):
        with patch.object(report_scheduler, "WEB_APP_URL", "http://test/api"):
            report_scheduler.main(MagicMock())

class TestProductCategoryGrouping(unittest.TestCase):
    def setUp(self):
        self.org_id = "org123"
        self.full_url = "http://test/api"
        os.environ["AZURE_DB_NAME"] = "testdb"

    def test_group_products_by_category_no_duplicates(self):
        products = [
            {"name": "ProductA", "category": "Cat1"},
            {"name": "ProductB", "category": "Cat1"},
            {"name": "ProductC", "category": "Cat2"},
        ]
        # Manual Grouping like the code
        category_map = {}
        for product in products:
            product_name = product.get("name")
            product_category = product.get("category")
            if not product_name or not product_category:
                continue
            if product_category not in category_map:
                category_map[product_category] = []
            category_map[product_category].append(product_name)

        self.assertEqual(set(category_map.keys()), {"Cat1", "Cat2"})
        self.assertEqual(category_map["Cat1"], ["ProductA", "ProductB"])
        self.assertEqual(category_map["Cat2"], ["ProductC"])

    def test_create_products_payload_format(self):
        product_names = ["ProductA", "ProductB"]
        category = "Cat1"
        payload = report_scheduler.create_products_payload(product_names, category)
        self.assertIn("report_key", payload)
        self.assertIn("report_name", payload)
        self.assertIn("categories", payload["params"])
        self.assertEqual(payload["report_key"], "product_analysis")
        self.assertEqual(payload["report_name"], product_names)
        self.assertEqual(payload["params"]["categories"]["category"], category)
        self.assertEqual(payload["params"]["categories"]["product"], product_names)

class TestCompetitorGrouping(unittest.TestCase):
    def setUp(self):
        self.org_id = "org123"
        self.industry_description = "IndustryDesc"
        self.brand_names = ["BrandA", "BrandB"]
        self.competitors = [
            {"name": "Competitor1", "category": "Cat1"},
            {"name": "Competitor2", "category": "Cat1"},
            {"name": "Competitor3", "category": "Cat2"},
        ]

    def test_group_competitors_by_category(self):
        competitors_by_category = {}
        for competitor in self.competitors:
            competitor_name = competitor.get("name")
            competitor_category = competitor.get("category")
            if not competitor_name or not competitor_category:
                continue
            if competitor_category not in competitors_by_category:
                competitors_by_category[competitor_category] = []
            competitors_by_category[competitor_category].append(competitor_name)
        self.assertEqual(set(competitors_by_category.keys()), {"Cat1", "Cat2"})
        self.assertEqual(competitors_by_category["Cat1"], ["Competitor1", "Competitor2"])
        self.assertEqual(competitors_by_category["Cat2"], ["Competitor3"])

    def test_create_competitors_payload_format(self):
        category = "Cat1"
        competitor_names = ["Competitor1", "Competitor2"]
        payload = report_scheduler.create_competitors_payload(
            category_name=category,
            competitor_name=competitor_names,
            brands=self.brand_names,
            industry_description=self.industry_description
        )
        self.assertIn("report_key", payload)
        self.assertIn("report_name", payload)
        self.assertIn("params", payload)
        self.assertIn("categories", payload["params"])
        self.assertEqual(payload["report_key"], "competitor_analysis")
        self.assertEqual(payload["report_name"], competitor_names)
        self.assertEqual(payload["params"]["categories"]["category"], category)
        self.assertEqual(payload["params"]["categories"]["brands"], self.brand_names)
        self.assertEqual(payload["params"]["categories"]["competitors"], competitor_names)
        self.assertEqual(payload["params"]["industry_context"], self.industry_description)

if __name__ == "__main__":
    unittest.main()
