"""Camera-independent tracking and durable observation storage."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from width_observations import record_width_change


def box_iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


class ObservationTracker:
    """Match nearby frames one-to-one; no identity claim across camera sessions."""

    def __init__(self, timeout=5.0, min_iou=0.3):
        self.timeout = timeout
        self.min_iou = min_iou
        self.tracks = {}

    def observe(self, records, now):
        self.tracks = {key: value for key, value in self.tracks.items()
                       if now - value["seen"] <= self.timeout}
        candidates = []
        for index, record in enumerate(records):
            for key, track in self.tracks.items():
                if track["context"] == record["inspection_key"]:
                    score = box_iou(record["bbox"], track["bbox"])
                    if score >= self.min_iou:
                        candidates.append((score, index, key))
        assignments, used = {}, set()
        for _, index, key in sorted(candidates, reverse=True):
            if index not in assignments and key not in used:
                assignments[index] = key
                used.add(key)
        for index, record in enumerate(records):
            key = assignments.get(index, uuid.uuid4().hex)
            old = self.tracks.get(key, {})
            record["observation_id"] = key
            record["first_seen"] = old.get("first_seen", record["timestamp"])
            self.tracks[key] = {"bbox": record["bbox"], "seen": now,
                                "context": record["inspection_key"],
                                "first_seen": record["first_seen"]}


class ObservationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS observations (
                number INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS width_samples (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS width_samples_observation ON width_samples(observation_id, sequence)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, records):
        saved = []
        with self.connect() as db:
            for original in records:
                record = dict(original)
                key = record["observation_id"]
                old = db.execute("SELECT number, payload FROM observations WHERE observation_id=?", (key,)).fetchone()
                previous = {}
                if old:
                    number, payload = old
                    previous = json.loads(payload)
                    record["image_url"] = record.get("image_url") or previous.get("image_url")
                    record["first_seen"] = previous.get("first_seen", record["timestamp"])
                else:
                    number = db.execute("INSERT INTO observations(observation_id,payload) VALUES (?,?)", (key, "{}")).lastrowid
                record["damage_no"] = f"D-{number:02d}"
                if record.get("width_profile_version"):
                    record.update(record_width_change(record, previous))
                    db.execute("INSERT INTO width_samples(observation_id,payload) VALUES (?,?)",
                               (key, json.dumps(record["width_last_sample"], ensure_ascii=False)))
                db.execute("UPDATE observations SET payload=? WHERE observation_id=?",
                           (json.dumps(record, ensure_ascii=False), key))
                saved.append(record)
        return saved

    def records(self, limit=None):
        with self.connect() as db:
            sql = "SELECT payload FROM observations ORDER BY number DESC"
            rows = db.execute(sql if limit is None else sql + " LIMIT ?",
                              () if limit is None else (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def attach_image(self, records, image_url):
        # UPDATE only: a capture finishing after 'clear history' must not restore it.
        with self.connect() as db:
            for record in records:
                key = record["observation_id"]
                row = db.execute("SELECT payload FROM observations WHERE observation_id=?", (key,)).fetchone()
                if row:
                    payload = json.loads(row[0])
                    payload["image_url"] = image_url
                    db.execute("UPDATE observations SET payload=? WHERE observation_id=?",
                               (json.dumps(payload, ensure_ascii=False), key))

    def clear(self):
        with self.connect() as db:
            db.execute("DELETE FROM width_samples")
            db.execute("DELETE FROM observations")
