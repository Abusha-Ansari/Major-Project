"""
Hospital registry and per-round submission tracking for the federated-learning server.

Files managed
─────────────
server_storage/hospitals.json
    Permanent registry mapping lowercase keys to hospital entries.
    Entry schema: {name, key, password (plain text), token (UUID4), registered_at (epoch int)}

server_storage/round_state.json
    Transient per-round record of which hospitals have already submitted.
    Schema: { "<round_id>": {"submitted": [key, ...], "names": [name, ...]} }
    Cleared automatically when the admin triggers aggregation.
"""

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from config import PROJECT_ROOT

HOSPITALS_PATH   = PROJECT_ROOT / "server_storage" / "hospitals.json"
ROUND_STATE_PATH = PROJECT_ROOT / "server_storage" / "round_state.json"

_lock = threading.Lock()


# ── internal helpers ──────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(path)


def _key(name: str) -> str:
    """Normalise a hospital name to a lowercase alphanumeric key."""
    cleaned = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()
    )
    return cleaned.strip("_").lower() or "hospital"


# ── Hospital registry ─────────────────────────────────────────────────────────

def register_hospital(name: str, password: str) -> dict:
    """
    Register a new hospital.

    Stores ``password`` as plain text in hospitals.json.
    Returns the new registry entry ``{name, key, token, registered_at}`` (password excluded).
    Raises ``ValueError`` if the normalised name is already taken or if password is empty.
    """
    if not password:
        raise ValueError("Password cannot be empty.")
    k = _key(name)
    with _lock:
        reg = _load(HOSPITALS_PATH)
        if k in reg:
            raise ValueError(f"Hospital '{name}' is already registered.")
        entry: dict = {
            "name": name,
            "key":  k,
            "password": password,
            "token": str(uuid.uuid4()),
            "registered_at": int(time.time()),
        }
        reg[k] = entry
        _save(HOSPITALS_PATH, reg)
        # Return entry without exposing password
        return {k2: v for k2, v in entry.items() if k2 != "password"}


def login_hospital(name: str, password: str) -> dict:
    """
    Verify the hospital's password and return the registry entry (without password field).

    Raises ``ValueError`` if not registered or password is wrong.
    """
    k = _key(name)
    with _lock:
        reg = _load(HOSPITALS_PATH)
        entry = reg.get(k)
    if entry is None:
        raise ValueError(f"Hospital '{name}' is not registered. Please register first.")
    if entry.get("password") != password:
        raise ValueError("Incorrect password.")
    return {k2: v for k2, v in entry.items() if k2 != "password"}


def validate_token(hospital_name: str, token: str) -> bool:
    """Return ``True`` iff the (name, token) pair matches the registry.

    Does a direct lookup — does NOT require the hospital's password.
    This is used only to verify weight-submission Bearer tokens.
    """
    k = _key(hospital_name)
    with _lock:
        entry = _load(HOSPITALS_PATH).get(k)
    return entry is not None and entry.get("token") == token


def list_hospitals() -> list:
    """Return all registered hospitals sorted by registration time."""
    with _lock:
        reg = _load(HOSPITALS_PATH)
        return sorted(reg.values(), key=lambda h: h["registered_at"])


# ── Round-state (duplicate submission prevention) ─────────────────────────────

def has_submitted(round_id: str, hospital_key: str) -> bool:
    """Return ``True`` if *hospital_key* already submitted for *round_id*."""
    with _lock:
        state = _load(ROUND_STATE_PATH)
        return hospital_key in state.get(round_id, {}).get("submitted", [])


def mark_submitted(round_id: str, hospital_key: str, hospital_name: str) -> None:
    """Record that *hospital_key* submitted an update for *round_id*."""
    with _lock:
        state = _load(ROUND_STATE_PATH)
        bucket = state.setdefault(round_id, {"submitted": [], "names": []})
        if hospital_key not in bucket["submitted"]:
            bucket["submitted"].append(hospital_key)
            bucket["names"].append(hospital_name)
        _save(ROUND_STATE_PATH, state)


def clear_round_state(round_id: str) -> None:
    """Remove submission records for *round_id* — called after aggregation."""
    with _lock:
        state = _load(ROUND_STATE_PATH)
        state.pop(round_id, None)
        _save(ROUND_STATE_PATH, state)


def get_round_state(round_id: str) -> dict:
    """Return ``{submitted: [...keys...], names: [...names...]}`` for *round_id*."""
    with _lock:
        return _load(ROUND_STATE_PATH).get(round_id, {"submitted": [], "names": []})
