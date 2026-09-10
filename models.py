"""
Firebase Firestore Database Models (with robust In-Memory Fallback)
Role: Define helper classes for User, Upload, Report, PresentationSession, and HistoricalReport entities.
Uses Firestore as primary database, with an in-memory store fallback if Firestore API is offline or disabled.
"""

import os
import json
from datetime import datetime, timezone
import uuid
import logging
import threading

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Thread Lock for In-Memory Storage & State Fallback ───────────────────────
_store_lock = threading.RLock()

# ── In-Memory Storage Fallback ────────────────────────────────────────────────
_MEMORY_STORE = {
    "users": {},                 # id -> dict
    "uploads": {},               # id -> dict
    "reports": {},               # id -> dict
    "presentation_sessions": {}, # id -> dict
    "historical_reports": {},    # id -> dict
}

_use_firestore = True

# ── Firebase Admin SDK setup ─────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore as fs

_firebase_app = None
db = None

def _disable_firestore():
    global _use_firestore
    with _store_lock:
        _use_firestore = False

def _is_firestore_enabled():
    with _store_lock:
        return _use_firestore and db is not None

def _init_firebase():
    global _firebase_app, db
    if _firebase_app is not None:
        return

    cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', 'firebase-service-account.json')
    cred_json_str = os.getenv('FIREBASE_CREDENTIALS_JSON', '').strip()

    try:
        if cred_json_str:
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            print("[DB OK] Firebase initialized from FIREBASE_CREDENTIALS_JSON env var")
        elif os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            print(f"[DB OK] Firebase initialized from credentials file: {cred_path}")
        else:
            print("[DB WARN] Firebase credentials file not found. Using local in-memory database.")
            _disable_firestore()
            return

        _firebase_app = firebase_admin.initialize_app(cred)
        db = fs.client()
        print("[DB OK] Connected to Firebase Firestore")

    except Exception as e:
        print(f"[DB WARN] Firebase initialization error ({str(e)}). Fallback to in-memory database active.")
        _disable_firestore()

_init_firebase()


# ── Helper utilities ──────────────────────────────────────────────────────────

def _to_dict(doc_snapshot) -> dict | None:
    if doc_snapshot and doc_snapshot.exists:
        return doc_snapshot.to_dict()
    return None


def _serialize_dt(dt) -> str | None:
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt) if dt else None


# ─────────────────────────────────────────────────────────────────────────────
# User Model
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# User Model
# ─────────────────────────────────────────────────────────────────────────────

class User:
    """User class — maps to Firestore 'users' collection or in-memory fallback."""

    def __init__(self, id, uid=None, name="", email="", photo_url=None, provider="password", password_hash=None, created_at=None, updated_at=None):
        self.id = id
        self.uid = uid or id
        self.name = name
        self.email = email
        self.photo_url = photo_url
        self.provider = provider or "password"
        self.password_hash = password_hash
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    @staticmethod
    def get_by_email(email: str) -> "User | None":
        email = email.lower().strip()
        if _is_firestore_enabled():
            try:
                results = list(db.collection("users").where("email", "==", email).limit(1).stream())
                for doc in results:
                    d = doc.to_dict()
                    return User(
                        id=d.get("id"), uid=d.get("uid") or d.get("id"),
                        name=d.get("name"), email=d.get("email"),
                        photo_url=d.get("photo_url"), provider=d.get("provider", "password"),
                        password_hash=d.get("password_hash"),
                        created_at=d.get("created_at"), updated_at=d.get("updated_at")
                    )
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on get_by_email: {e}. Switching to in-memory store.")
                _disable_firestore()

        # In-memory search with thread lock
        with _store_lock:
            users_list = list(_MEMORY_STORE["users"].values())

        for u in users_list:
            if u.get("email") == email:
                return User(
                    id=u.get("id"), uid=u.get("uid") or u.get("id"),
                    name=u.get("name"), email=u.get("email"),
                    photo_url=u.get("photo_url"), provider=u.get("provider", "password"),
                    password_hash=u.get("password_hash"),
                    created_at=u.get("created_at"), updated_at=u.get("updated_at")
                )
        return None

    @staticmethod
    def get_by_id(user_id: str) -> "User | None":
        if _is_firestore_enabled():
            try:
                doc = db.collection("users").document(user_id).get()
                d = _to_dict(doc)
                if d:
                    return User(
                        id=d.get("id"), uid=d.get("uid") or d.get("id"),
                        name=d.get("name"), email=d.get("email"),
                        photo_url=d.get("photo_url"), provider=d.get("provider", "password"),
                        password_hash=d.get("password_hash"),
                        created_at=d.get("created_at"), updated_at=d.get("updated_at")
                    )
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on get_by_id: {e}. Switching to in-memory store.")
                _disable_firestore()

        # In-memory lookup with thread lock
        with _store_lock:
            u = _MEMORY_STORE["users"].get(user_id)

        if u:
            return User(
                id=u.get("id"), uid=u.get("uid") or u.get("id"),
                name=u.get("name"), email=u.get("email"),
                photo_url=u.get("photo_url"), provider=u.get("provider", "password"),
                password_hash=u.get("password_hash"),
                created_at=u.get("created_at"), updated_at=u.get("updated_at")
            )
        return None

    @staticmethod
    def create(name: str, email: str, photo_url: str = None, provider: str = "password", password_hash: str = None) -> "User":
        user_id = str(uuid.uuid4())
        return User.create_with_id(user_id, name, email, photo_url, provider, password_hash)

    @staticmethod
    def create_with_id(user_id: str, name: str, email: str, photo_url: str = None, provider: str = "password", password_hash: str = None) -> "User":
        now = datetime.now(timezone.utc)
        doc = {
            "id": user_id,
            "uid": user_id,
            "name": name.strip() if name else "",
            "email": email.lower().strip() if email else "",
            "photo_url": photo_url,
            "provider": provider or "password",
            "password_hash": password_hash,
            "created_at": now,
            "updated_at": now,
        }
        with _store_lock:
            _MEMORY_STORE["users"][user_id] = doc

        if _is_firestore_enabled():
            try:
                db.collection("users").document(user_id).set(doc, merge=True)
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on User.create_with_id: {e}")
                _disable_firestore()

        return User(
            id=user_id, uid=user_id, name=name, email=email,
            photo_url=photo_url, provider=provider, password_hash=password_hash,
            created_at=now, updated_at=now
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "uid": self.uid,
            "name": self.name,
            "email": self.email,
            "photo_url": self.photo_url,
            "provider": self.provider,
            "created_at": _serialize_dt(self.created_at),
            "updated_at": _serialize_dt(self.updated_at),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Upload Model
# ─────────────────────────────────────────────────────────────────────────────

class Upload:
    """Upload class — maps to Firestore 'uploads' collection or in-memory fallback."""

    def __init__(self, id, filename, mime_type, file_path, user_id, created_at=None):
        self.id = id
        self.filename = filename
        self.mime_type = mime_type
        self.file_path = file_path
        self.user_id = user_id
        self.created_at = created_at or datetime.now(timezone.utc)

    @staticmethod
    def create(filename: str, mime_type: str, file_path: str, user_id: str) -> "Upload":
        upload_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doc = {
            "id": upload_id,
            "filename": filename,
            "mime_type": mime_type,
            "file_path": file_path,
            "user_id": user_id,
            "created_at": now,
        }
        with _store_lock:
            _MEMORY_STORE["uploads"][upload_id] = doc

        if _is_firestore_enabled():
            try:
                db.collection("uploads").document(upload_id).set(doc)
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on Upload.create: {e}")
                _disable_firestore()

        return Upload(
            id=upload_id, filename=filename, mime_type=mime_type,
            file_path=file_path, user_id=user_id, created_at=now
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "user_id": self.user_id,
            "created_at": _serialize_dt(self.created_at),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Report Model
# ─────────────────────────────────────────────────────────────────────────────

class Report:
    """Report class — maps to Firestore 'reports' collection or in-memory fallback."""

    def __init__(self, id, report_json, report_type, user_id, upload_id=None,
                 created_at=None, updated_at=None):
        self.id = id
        self.report_json = report_json
        self.report_type = report_type
        self.user_id = user_id
        self.upload_id = upload_id
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    @staticmethod
    def create(report_json: dict, report_type: str, user_id: str,
               upload_id: str | None = None) -> "Report":
        report_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doc = {
            "id": report_id,
            "report_json": report_json,
            "report_type": report_type,
            "user_id": user_id,
            "upload_id": upload_id,
            "created_at": now,
            "updated_at": now,
        }
        with _store_lock:
            _MEMORY_STORE["reports"][report_id] = doc

        if _is_firestore_enabled():
            try:
                db.collection("reports").document(report_id).set(doc)
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on Report.create: {e}")
                _disable_firestore()

        return Report(
            id=report_id, report_json=report_json, report_type=report_type,
            user_id=user_id, upload_id=upload_id, created_at=now, updated_at=now
        )

    @staticmethod
    def get_by_user(user_id: str) -> list["Report"]:
        if _is_firestore_enabled():
            try:
                results = list(
                    db.collection("reports")
                    .where("user_id", "==", user_id)
                    .stream()
                )
                reports = []
                for doc in results:
                    d = doc.to_dict()
                    reports.append(Report(
                        id=d.get("id"), report_json=d.get("report_json"),
                        report_type=d.get("report_type"), user_id=d.get("user_id"),
                        upload_id=d.get("upload_id"), created_at=d.get("created_at"),
                        updated_at=d.get("updated_at")
                    ))
                return reports
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on Report.get_by_user: {e}")
                _disable_firestore()

        # In-memory search with thread lock
        with _store_lock:
            reports_list = list(_MEMORY_STORE["reports"].values())

        reports = []
        for r in reports_list:
            if r.get("user_id") == user_id:
                reports.append(Report(
                    id=r.get("id"), report_json=r.get("report_json"),
                    report_type=r.get("report_type"), user_id=r.get("user_id"),
                    upload_id=r.get("upload_id"), created_at=r.get("created_at"),
                    updated_at=r.get("updated_at")
                ))
        return sorted(reports, key=lambda x: str(x.created_at), reverse=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "report_type": self.report_type,
            "report_json": self.report_json,
            "user_id": self.user_id,
            "upload_id": self.upload_id,
            "created_at": _serialize_dt(self.created_at),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PresentationSession Model
# ─────────────────────────────────────────────────────────────────────────────

class PresentationSession:
    """PresentationSession — maps to Firestore 'presentation_sessions' collection."""

    def __init__(self, id, user_id, topic, status, started_at=None, ended_at=None, metrics=None):
        self.id = id
        self.user_id = user_id
        self.topic = topic
        self.status = status
        self.started_at = started_at or datetime.now(timezone.utc)
        self.ended_at = ended_at or datetime.now(timezone.utc)
        self.metrics = metrics or {
            "eye_contact_scores": [],
            "posture_scores": [],
            "wpm_history": [],
            "fillers_detected": 0,
            "transcripts": [],
            "interruptions": [],
            "confidence_scores": [],
            "vocal_sentiment_scores": [],
        }

    @staticmethod
    def create(user_id: str, topic: str) -> "PresentationSession":
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        default_metrics = {
            "eye_contact_scores": [],
            "posture_scores": [],
            "wpm_history": [],
            "fillers_detected": 0,
            "transcripts": [],
            "interruptions": [],
            "confidence_scores": [],
            "vocal_sentiment_scores": [],
        }
        doc = {
            "id": session_id,
            "user_id": user_id,
            "topic": topic,
            "status": "STREAMING",
            "started_at": now,
            "ended_at": now,
            "metrics": default_metrics,
        }
        with _store_lock:
            _MEMORY_STORE["presentation_sessions"][session_id] = doc

        if _is_firestore_enabled():
            try:
                db.collection("presentation_sessions").document(session_id).set(doc)
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on PresentationSession.create: {e}")
                _disable_firestore()

        return PresentationSession(
            id=session_id, user_id=user_id, topic=topic,
            status="STREAMING", started_at=now, ended_at=now, metrics=default_metrics
        )

    @staticmethod
    def get_by_id(session_id: str) -> "PresentationSession | None":
        # Fast path: check active in-memory sessions first to prevent socket blocking
        with _store_lock:
            s = _MEMORY_STORE["presentation_sessions"].get(session_id)

        if s:
            return PresentationSession(
                id=s.get("id"), user_id=s.get("user_id"),
                topic=s.get("topic"), status=s.get("status"),
                started_at=s.get("started_at"), ended_at=s.get("ended_at"),
                metrics=s.get("metrics")
            )

        if _is_firestore_enabled():
            try:
                doc = db.collection("presentation_sessions").document(session_id).get()
                d = _to_dict(doc)
                if d:
                    sess = PresentationSession(
                        id=d.get("id"), user_id=d.get("user_id"),
                        topic=d.get("topic"), status=d.get("status"),
                        started_at=d.get("started_at"), ended_at=d.get("ended_at"),
                        metrics=d.get("metrics")
                    )
                    with _store_lock:
                        _MEMORY_STORE["presentation_sessions"][session_id] = d
                    return sess
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on PresentationSession.get_by_id: {e}")
                _disable_firestore()

        return None

    def update_metrics(self, key: str, value) -> None:
        if self.metrics and key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key].append(value)

        with _store_lock:
            if self.id in _MEMORY_STORE["presentation_sessions"]:
                _MEMORY_STORE["presentation_sessions"][self.id]["metrics"] = self.metrics

        # Note: during high-frequency real-time streaming, in-memory store prevents
        # socket event-loop starvation. Firestore is updated safely in background.
        if _is_firestore_enabled():
            try:
                ref = db.collection("presentation_sessions").document(self.id)
                ref.update({f"metrics.{key}": fs.ArrayUnion([value])})
            except Exception as e:
                logger.debug(f"[DB] Firestore non-blocking metric update notice: {e}")

    def increment_metric(self, key: str, val: int = 1) -> None:
        if self.metrics and key in self.metrics:
            self.metrics[key] = self.metrics.get(key, 0) + val

        with _store_lock:
            if self.id in _MEMORY_STORE["presentation_sessions"]:
                _MEMORY_STORE["presentation_sessions"][self.id]["metrics"] = self.metrics

        if _is_firestore_enabled():
            try:
                ref = db.collection("presentation_sessions").document(self.id)
                ref.update({f"metrics.{key}": fs.Increment(val)})
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore increment_metric error: {e}")
                _disable_firestore()

    def update_status(self, new_status: str) -> None:
        self.status = new_status
        now = datetime.now(timezone.utc)
        with _store_lock:
            if self.id in _MEMORY_STORE["presentation_sessions"]:
                _MEMORY_STORE["presentation_sessions"][self.id]["status"] = new_status
                _MEMORY_STORE["presentation_sessions"][self.id]["ended_at"] = now

        if _is_firestore_enabled():
            try:
                db.collection("presentation_sessions").document(self.id).update({
                    "status": new_status,
                    "ended_at": now,
                })
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore update_status error: {e}")
                _disable_firestore()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "topic": self.topic,
            "status": self.status,
            "started_at": _serialize_dt(self.started_at),
            "ended_at": _serialize_dt(self.ended_at),
            "metrics": self.metrics,
        }


# ─────────────────────────────────────────────────────────────────────────────
# HistoricalReport Model
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalReport:
    """HistoricalReport — maps to Firestore 'historical_reports' collection."""

    def __init__(self, id, session_id, user_id, topic, report_json, created_at=None):
        self.id = id
        self.session_id = session_id
        self.user_id = user_id
        self.topic = topic
        self.report_json = report_json
        self.created_at = created_at or datetime.now(timezone.utc)

    @staticmethod
    def create(session_id: str, user_id: str, topic: str, report_json: dict) -> "HistoricalReport":
        report_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        doc = {
            "id": report_id,
            "session_id": session_id,
            "user_id": user_id,
            "topic": topic.strip().lower(),
            "report_json": report_json,
            "created_at": now,
        }
        with _store_lock:
            _MEMORY_STORE["historical_reports"][report_id] = doc

        if _is_firestore_enabled():
            try:
                db.collection("historical_reports").document(report_id).set(doc)
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on HistoricalReport.create: {e}")
                _disable_firestore()

        return HistoricalReport(
            id=report_id, session_id=session_id, user_id=user_id,
            topic=topic, report_json=report_json, created_at=now
        )

    @staticmethod
    def get_by_user_and_topic(user_id: str, topic: str) -> list["HistoricalReport"]:
        topic_lower = topic.strip().lower()
        if _is_firestore_enabled():
            try:
                results = list(
                    db.collection("historical_reports")
                    .where("user_id", "==", user_id)
                    .where("topic", "==", topic_lower)
                    .stream()
                )
                reports = []
                for doc in results:
                    d = doc.to_dict()
                    reports.append(HistoricalReport(
                        id=d.get("id"), session_id=d.get("session_id"),
                        user_id=d.get("user_id"), topic=d.get("topic"),
                        report_json=d.get("report_json"), created_at=d.get("created_at")
                    ))
                return reports
            except Exception as e:
                logger.warning(f"[DB FALLBACK] Firestore error on get_by_user_and_topic: {e}")
                _disable_firestore()

        with _store_lock:
            hr_list = list(_MEMORY_STORE["historical_reports"].values())

        reports = []
        for hr in hr_list:
            if hr.get("user_id") == user_id and hr.get("topic") == topic_lower:
                reports.append(HistoricalReport(
                    id=hr.get("id"), session_id=hr.get("session_id"),
                    user_id=hr.get("user_id"), topic=hr.get("topic"),
                    report_json=hr.get("report_json"), created_at=hr.get("created_at")
                ))
        return sorted(reports, key=lambda x: str(x.created_at), reverse=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "topic": self.topic,
            "report_json": self.report_json,
            "created_at": _serialize_dt(self.created_at),
        }
