"""Transactional local evidence journal, upload outbox and request/response audit."""

import json
import sqlite3
import time
from pathlib import Path

from memory.base import MemoryRecord, RunContext


class MemoryStore:
    def __init__(self, root: Path, context: RunContext):
        self.context = context
        directory = Path(root) / context.audience
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "memory.sqlite3")
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS records (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id TEXT NOT NULL UNIQUE,
                event_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL UNIQUE,
                record TEXT NOT NULL,
                superseded INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS outbox (
                source_id TEXT PRIMARY KEY REFERENCES records(source_id),
                body TEXT NOT NULL,
                ingestion_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                operation TEXT NOT NULL,
                payload TEXT NOT NULL
            );
        """)

    def append(self, record: MemoryRecord, source_id: str, body: dict | None = None) -> None:
        encoded = record.model_dump_json()
        with self.db:
            existing = self.db.execute(
                "SELECT record FROM records WHERE evidence_id=?", (record.evidence.id,)
            ).fetchone()
            if existing:
                if existing["record"] != encoded:
                    raise ValueError("Evidence IDs are immutable; write a new ID with supersedes")
                return
            self.db.execute(
                "INSERT INTO records(evidence_id,event_id,source_id,record) VALUES(?,?,?,?)",
                (record.evidence.id, record.event_id, source_id, encoded),
            )
            if body is not None:
                self.db.execute(
                    "INSERT INTO outbox(source_id,body) VALUES(?,?)",
                    (source_id, json.dumps(body, sort_keys=True)),
                )
            for evidence_id in record.supersedes:
                self.db.execute(
                    "UPDATE records SET superseded=1 WHERE evidence_id=?", (evidence_id,)
                )

    def by_evidence(self, evidence_id: str) -> MemoryRecord | None:
        row = self.db.execute(
            "SELECT record FROM records WHERE evidence_id=?", (evidence_id,)
        ).fetchone()
        return MemoryRecord.model_validate_json(row["record"]) if row else None

    def eligible(self) -> list[sqlite3.Row]:
        return self.db.execute("""
            SELECT records.*, COALESCE(outbox.status, 'local') AS status FROM records
            LEFT JOIN outbox USING(source_id) WHERE superseded=0 ORDER BY seq
        """).fetchall()

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute("""
            SELECT outbox.* FROM outbox JOIN records USING(source_id)
            WHERE outbox.status != 'completed' ORDER BY records.seq
        """).fetchall()

    def progress(self, source_id: str, **fields) -> None:
        if not fields or not set(fields) <= {"ingestion_id", "status", "attempts", "error"}:
            raise ValueError("Invalid outbox fields")
        assignments = ",".join(f"{field}=?" for field in fields)
        with self.db:
            self.db.execute(
                f"UPDATE outbox SET {assignments} WHERE source_id=?",
                (*fields.values(), source_id),
            )

    def audit(self, operation: str, payload: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO audit(ts,operation,payload) VALUES(?,?,?)",
                (time.time(), operation, json.dumps(payload, sort_keys=True)),
            )

    def close(self) -> None:
        self.db.close()
