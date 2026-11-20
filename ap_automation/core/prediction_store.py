"""
Predictions awaiting confirmation, persisted in SQLite.

A prediction is made on one request and confirmed on another, so the API has to remember it in
between. Holding it only in process memory meant a restart (or an idle spin-down on a free host)
made /confirm return 404 mid-demo. The path comes from settings.predictions_db_path, so tests and
the demo can point it somewhere disposable.

Note: on hosts with an ephemeral filesystem (e.g. Render's free tier) the file is lost when the
instance is replaced - mount a persistent disk, or run the demo locally.
"""

from __future__ import annotations

import time

from ap_automation.core import database
from ap_automation.core.config import settings
from ap_automation.core.models import Prediction, RoutingDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id      TEXT PRIMARY KEY,
    created REAL NOT NULL,
    posted  INTEGER NOT NULL DEFAULT 0,
    body    TEXT NOT NULL
)
"""


class PredictionStore:
    def __init__(self, limit: int | None = None):
        self._limit = limit

    @property
    def limit(self) -> int:
        return self._limit or settings.max_tracked_predictions

    def _conn(self):
        cm = database._get_conn(settings.predictions_db_path)
        return cm

    def _ensure(self, conn) -> None:
        conn.execute(_SCHEMA)

    def put(self, pred: Prediction) -> None:
        with self._conn() as conn:
            self._ensure(conn)
            conn.execute(
                "INSERT OR REPLACE INTO predictions (id, created, posted, body) VALUES (?, ?, 0, ?)",
                (str(pred.prediction_id), time.time(), pred.model_dump_json()),
            )
            conn.execute(
                "DELETE FROM predictions WHERE created < "
                "COALESCE((SELECT created FROM predictions ORDER BY created DESC LIMIT 1 OFFSET ?), 0)",
                (self.limit,),
            )

    def get(self, pid: str) -> dict | None:
        with self._conn() as conn:
            self._ensure(conn)
            row = conn.execute("SELECT posted, body FROM predictions WHERE id = ?", (pid,)).fetchone()
        if row is None:
            return None
        return {"pred": Prediction.model_validate_json(row["body"]), "posted": bool(row["posted"])}

    def pending_auto_post(self, limit: int = 1000) -> list[tuple[Prediction, float]]:
        """
        Predictions the governed gate cleared (final_routing == auto_post) that have not been posted, newest first,
        each with the time it was made. Read-only: nothing is claimed or changed.
        """
        out: list[tuple[Prediction, float]] = []
        with self._conn() as conn:
            self._ensure(conn)
            rows = conn.execute(
                "SELECT created, body FROM predictions WHERE posted = 0 AND body LIKE ? "
                "ORDER BY created DESC LIMIT ?",
                ('%"final_routing":"auto_post"%', limit),
            ).fetchall()
        for r in rows:
            pred = Prediction.model_validate_json(r["body"])
            if pred.final_routing == RoutingDecision.AUTO_POST:     # the LIKE is only a pre-filter
                out.append((pred, r["created"]))
        return out

    def claim(self, pid: str) -> Prediction | None:
        """Atomically mark as posted. Returns None if it is missing or was already posted."""
        with self._conn() as conn:
            self._ensure(conn)
            cur = conn.execute("UPDATE predictions SET posted = 1 WHERE id = ? AND posted = 0", (pid,))
            if cur.rowcount != 1:
                return None
            row = conn.execute("SELECT body FROM predictions WHERE id = ?", (pid,)).fetchone()
        return Prediction.model_validate_json(row["body"])

    def release(self, pid: str) -> None:
        with self._conn() as conn:
            self._ensure(conn)
            conn.execute("UPDATE predictions SET posted = 0 WHERE id = ?", (pid,))

    def return_to_review(self, pid: str, reason: str) -> None:
        """
        Put a prediction back in the human-review queue: unposted, and no longer cleared for auto-post.

        Used when the ERP did not accept a post. It can only make the decision stricter (never auto_post), and it
        keeps the ERP's error as the routing reason so a reviewer sees why. A human can then /confirm it.
        """
        with self._conn() as conn:
            self._ensure(conn)
            row = conn.execute("SELECT body FROM predictions WHERE id = ?", (pid,)).fetchone()
            if row is None:
                return
            pred = Prediction.model_validate_json(row["body"])
            pred.final_routing = RoutingDecision.HUMAN_REVIEW
            pred.final_routing_reason = reason
            pred.provenance = {**pred.provenance, "final_routing": RoutingDecision.HUMAN_REVIEW.value,
                               "final_routing_reason": reason}
            conn.execute("UPDATE predictions SET posted = 0, body = ? WHERE id = ?", (pred.model_dump_json(), pid))

    def clear(self) -> None:
        with self._conn() as conn:
            self._ensure(conn)
            conn.execute("DELETE FROM predictions")

    def count(self) -> int:
        with self._conn() as conn:
            self._ensure(conn)
            return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
