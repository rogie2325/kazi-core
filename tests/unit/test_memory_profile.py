"""
Unit tests for kazi/memory/profile.py — UserProfile CRUD, system preamble,
path sanitisation, and traversal guard.  Pure file I/O — no API keys needed.
"""
import json
from pathlib import Path

import pytest

from kazi.memory.profile import UserProfile


@pytest.fixture
def profile(tmp_path):
    return UserProfile(storage_dir=str(tmp_path / "profiles"))


# ── load ──────────────────────────────────────────────────────────────────────

def test_load_returns_empty_dict_for_unknown_user(profile):
    assert profile.load("nobody") == {}


def test_load_returns_stored_data(profile):
    profile.save("alice", {"role": "engineer"})
    assert profile.load("alice") == {"role": "engineer"}


def test_load_returns_empty_on_corrupt_file(profile, tmp_path):
    p = Path(str(tmp_path / "profiles")) / "alice.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("NOT JSON", encoding="utf-8")
    assert profile.load("alice") == {}


# ── save ──────────────────────────────────────────────────────────────────────

def test_save_writes_json_file(profile, tmp_path):
    profile.save("bob", {"city": "Berlin"})
    raw = (tmp_path / "profiles" / "bob.json").read_text()
    assert json.loads(raw) == {"city": "Berlin"}


def test_save_overwrites_existing_profile(profile):
    profile.save("carol", {"x": 1})
    profile.save("carol", {"x": 99})
    assert profile.load("carol")["x"] == 99


# ── update ────────────────────────────────────────────────────────────────────

def test_update_merges_into_existing(profile):
    profile.save("dan", {"a": 1, "b": 2})
    profile.update("dan", {"b": 99, "c": 3})
    data = profile.load("dan")
    assert data == {"a": 1, "b": 99, "c": 3}


def test_update_creates_profile_if_absent(profile):
    profile.update("eve", {"role": "admin"})
    assert profile.load("eve") == {"role": "admin"}


# ── get ───────────────────────────────────────────────────────────────────────

def test_get_returns_fact(profile):
    profile.save("frank", {"timezone": "UTC"})
    assert profile.get("frank", "timezone") == "UTC"


def test_get_returns_default_when_key_absent(profile):
    profile.save("frank", {})
    assert profile.get("frank", "missing", "fallback") == "fallback"


def test_get_returns_none_default_for_unknown_user(profile):
    assert profile.get("ghost", "key") is None


# ── delete_fact ───────────────────────────────────────────────────────────────

def test_delete_fact_removes_key(profile):
    profile.save("grace", {"a": 1, "b": 2})
    profile.delete_fact("grace", "a")
    assert "a" not in profile.load("grace")
    assert profile.load("grace")["b"] == 2


def test_delete_fact_is_noop_for_missing_key(profile):
    profile.save("grace", {"b": 2})
    profile.delete_fact("grace", "nonexistent")  # should not raise


# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_removes_profile_file(profile, tmp_path):
    profile.save("hank", {"x": 1})
    profile.delete("hank")
    assert not (tmp_path / "profiles" / "hank.json").exists()


def test_delete_is_noop_for_unknown_user(profile):
    profile.delete("nobody")  # should not raise


# ── list_users ────────────────────────────────────────────────────────────────

def test_list_users_returns_all_stored_users(profile):
    profile.save("alice", {})
    profile.save("bob", {})
    users = profile.list_users()
    assert "alice" in users
    assert "bob" in users


def test_list_users_empty_when_no_profiles(profile):
    assert profile.list_users() == []


# ── as_system_preamble ────────────────────────────────────────────────────────

def test_as_system_preamble_returns_none_for_empty_profile(profile):
    assert profile.as_system_preamble("nobody") is None


def test_as_system_preamble_includes_facts(profile):
    profile.save("iris", {"role": "scientist", "timezone": "UTC"})
    preamble = profile.as_system_preamble("iris")
    assert preamble is not None
    assert "role" in preamble
    assert "scientist" in preamble
    assert "timezone" in preamble
    assert "iris" in preamble


def test_as_system_preamble_is_multiline(profile):
    profile.save("ivan", {"a": 1, "b": 2})
    preamble = profile.as_system_preamble("ivan")
    assert "\n" in preamble


# ── _path sanitisation ────────────────────────────────────────────────────────

def test_path_sanitises_special_chars(profile, tmp_path):
    profile.save("user/with/slashes", {"x": 1})
    # Should not create nested directories — slash is replaced
    files = list((tmp_path / "profiles").glob("*.json"))
    assert len(files) == 1
    assert "/" not in files[0].name


def test_path_sanitises_null_bytes(profile):
    profile.save("user\x00null", {"x": 1})
    assert profile.load("user\x00null") == {"x": 1}


def test_path_caps_at_128_chars(profile):
    long_id = "a" * 200
    profile.save(long_id, {"x": 1})
    assert profile.load(long_id) == {"x": 1}


def test_path_empty_user_id_becomes_anonymous(profile):
    profile.save("", {"x": 1})
    assert profile.load("") == {"x": 1}


def test_path_traversal_via_symlink_raises(tmp_path):
    """Symlink pointing outside the storage dir is caught by the traversal guard."""
    import os
    storage = tmp_path / "profiles"
    storage.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")

    # Create a symlink inside the storage dir that resolves outside it
    symlink_name = "traversal_target"
    symlink_path = storage / f"{symlink_name}.json"
    os.symlink(str(outside), str(symlink_path))

    profile = UserProfile(storage_dir=str(storage))
    with pytest.raises(ValueError, match="path traversal"):
        profile.load(symlink_name)
