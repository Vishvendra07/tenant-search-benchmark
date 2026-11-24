import sqlite3
import unittest
from project import demo, search, setup


class SearchTests(unittest.TestCase):
    def test_tenant_scope_applies_before_rank_and_unknown_tenant_gets_no_hits(self):
        with sqlite3.connect(":memory:") as db:
            setup(db)
            db.execute("INSERT INTO docs VALUES ('harbor','H-secret','bank funding statement')")
            for mode in ("keyword", "vector"):
                rows = search(db, "north", "bank funding statement", mode)
                self.assertTrue(rows)
                self.assertTrue(all(row.startswith("N-") for row in rows))
                self.assertEqual(search(db, "unknown", "funding", mode), [])

    def test_aliases_beat_the_baseline_on_named_fixtures(self):
        report = demo()
        self.assertGreater(report["vector"]["recall_at_3"], report["keyword"]["recall_at_3"])
