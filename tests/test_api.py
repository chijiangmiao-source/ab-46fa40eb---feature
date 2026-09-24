"""End-to-end HTTP API tests against the real standard-library server."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import build_server


class ServerFixture:
    def __init__(self):
        self.server = build_server("127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def post(self, payload, raw=False):
        data = payload if raw else json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/v1/invert",
            data=data, headers={"Content-Type": "application/json"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
            return resp.status, json.loads(resp.read())


BODY = {
    "target": 100_000,
    "tolerance": 500,
    "components": [
        {"id": "monomer-A", "mass": 18_013, "min": 0, "max": 20},
        {"id": "adduct-Na", "mass": 22_990, "min": 0, "max": 10},
        {"id": "monomer-B", "mass": 44_000, "min": 0, "max": 5},
    ],
}


class ApiTests(unittest.TestCase):
    def test_health(self):
        with ServerFixture() as fx:
            status, body = fx.get("/healthz")
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "ok")

    def test_optimal_inversion(self):
        with ServerFixture() as fx:
            status, body = fx.post(BODY)
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "optimal")
            self.assertTrue(body["within_tolerance"])
            self.assertGreaterEqual(len(body["explanations"]), 1)
            expl = body["explanations"][0]
            # Every figure must be an integer and recompute exactly.
            self.assertIsInstance(expl["total_mass"], int)
            self.assertIsInstance(expl["error"], int)
            self.assertEqual(
                expl["recomputed_total_mass"],
                sum(c["count"] * c["mass"] for c in expl["counts"]))
            self.assertEqual(expl["recomputed_total_mass"], expl["total_mass"])
            self.assertEqual(expl["error"], expl["total_mass"] - BODY["target"])
            self.assertLessEqual(expl["absolute_error"], BODY["tolerance"])
            self.assertIn("unique", body)
            self.assertIsInstance(body["num_optimal_explanations"], int)
            # component ids present in declared order
            self.assertEqual(set(body["component_order"]),
                             {"monomer-A", "adduct-Na", "monomer-B"})
            for c in expl["counts"]:
                self.assertGreaterEqual(c["count"], c["min"])
                self.assertLessEqual(c["count"], c["max"])

    def test_non_unique_reports_alternative(self):
        # masses 4 and 6, target 11, tolerance 1 -> distance 1 both sides,
        # several minimum-count explanations.
        payload = {
            "target": 11, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 4, "min": 0, "max": 6},
                {"id": "b", "mass": 6, "min": 0, "max": 6},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 200)
            self.assertFalse(body["unique"])
            self.assertGreaterEqual(body["num_optimal_explanations"], 2)
            self.assertIsNotNone(body["alternative_witness"])

    def test_unsatisfiable_returns_witnesses(self):
        payload = {
            "target": 50, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 7, "min": 1, "max": 4},
                {"id": "b", "mass": 13, "min": 1, "max": 4},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 200)
            self.assertEqual(body["status"], "unsatisfiable")
            self.assertFalse(body["within_tolerance"])
            self.assertIsNotNone(body["nearest_below"])
            self.assertIsNotNone(body["nearest_above"])
            self.assertLessEqual(body["nearest_below"]["total_mass"], 50)
            self.assertGreaterEqual(body["nearest_above"]["total_mass"], 50)

    def test_structured_validation_errors_are_locatable(self):
        payload = {
            "target": -3,
            "tolerance": "x",
            "components": [
                {"id": "a", "mass": 1.5, "min": 5, "max": 2},
                {"id": "a", "mass": 0},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            paths = {e["path"]: e["code"] for e in body["errors"]}
            self.assertIn("/target", paths)
            self.assertIn("/tolerance", paths)
            self.assertIn("/components/0/mass", paths)
            self.assertIn("/components/0/min", paths)
            self.assertIn("/components/1/mass", paths)
            self.assertIn("/components/1/id", paths)

    def test_wrong_component_count(self):
        payload = {"target": 10, "tolerance": 1,
                   "components": [{"id": "a", "mass": 2}]}
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            self.assertEqual(body["errors"][0]["path"], "/components")

    def test_invalid_json(self):
        with ServerFixture() as fx:
            status, body = fx.post(b"{not json", raw=True)
            self.assertEqual(status, 400)
            self.assertEqual(body["error"]["code"], "invalid_json")

    def test_integer_only_enforcement(self):
        payload = {
            "target": 100.25, "tolerance": 1,
            "components": [
                {"id": "a", "mass": 4},
                {"id": "b", "mass": 6},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            self.assertIn("/target",
                          [e["path"] for e in body["errors"]])

    def test_bound_limit_one_million(self):
        payload = {
            "target": 10**12, "tolerance": 10**6,
            "components": [
                {"id": "a", "mass": 123_456_789, "max": 1_000_001},
                {"id": "b", "mass": 987_654_321},
            ],
        }
        with ServerFixture() as fx:
            status, body = fx.post(payload)
            self.assertEqual(status, 400)
            self.assertIn("/components/0/max",
                          [e["path"] for e in body["errors"]])


if __name__ == "__main__":
    unittest.main()
