import datetime as dt
import unittest

from common.data import DataLabel
from common.date_range import DateRange
from scraping.scraper import ScrapeConfig
from scraping.x.twikit_scraper import (
    build_twitter_search_query,
    extract_status_id_from_url,
)


class TestTwikitScraperHelpers(unittest.TestCase):
    def test_extract_status_id(self):
        self.assertEqual(
            extract_status_id_from_url("https://x.com/foo/status/1234567890123456789"),
            "1234567890123456789",
        )
        self.assertEqual(
            extract_status_id_from_url("https://twitter.com/bar/status/99"),
            "99",
        )
        self.assertIsNone(extract_status_id_from_url("https://x.com/foo"))
        self.assertIsNone(extract_status_id_from_url(""))

    def test_build_search_query_labels_and_dates(self):
        dr = DateRange(
            start=dt.datetime(2024, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc),
            end=dt.datetime(2024, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc),
        )
        cfg = ScrapeConfig(
            entity_limit=10,
            date_range=dr,
            labels=[
                DataLabel(value="@alice"),
                DataLabel(value="#btc"),
            ],
        )
        q = build_twitter_search_query(cfg)
        self.assertIn("since:2024-01-01", q)
        self.assertIn("until:2024-01-02", q)
        self.assertIn("from:alice", q)
        self.assertIn("#btc", q)

    def test_build_search_query_no_labels(self):
        dr = DateRange(
            start=dt.datetime(2024, 6, 1, 12, 0, 0, tzinfo=dt.timezone.utc),
            end=dt.datetime(2024, 6, 1, 13, 0, 0, tzinfo=dt.timezone.utc),
        )
        cfg = ScrapeConfig(entity_limit=5, date_range=dr, labels=None)
        q = build_twitter_search_query(cfg)
        self.assertTrue(q.endswith(" e"))


if __name__ == "__main__":
    unittest.main()
