"""Künye (provenance) modülünün testleri.

Künye üretimi HİÇBİR koşulda patlamamalı: git kurulu olmasa, klasör depo
olmasa, dosya bulunmasa bile bir sonuç dönmeli. Aksi halde "künye alınamadı"
diye bütün eğitim çöker - kaydedilmesi gereken sonuç da kaydedilmez.
"""

from __future__ import annotations

from tcm.provenance import (
    UNKNOWN,
    config_digest,
    file_digest,
    git_hash,
    package_versions,
    relative_path,
    run_stamp,
)


class TestGitHash:
    def test_returns_a_string_and_a_flag(self):
        commit, dirty = git_hash()
        assert isinstance(commit, str) and commit
        assert isinstance(dirty, bool)

    def test_outside_a_repository_it_does_not_crash(self, tmp_path):
        commit, dirty = git_hash(tmp_path)
        assert commit == UNKNOWN
        assert dirty is False


class TestDigests:
    def test_same_bytes_give_the_same_digest(self, tmp_path):
        a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
        a.write_text("wear_limit_um: 300\n", encoding="utf-8")
        b.write_text("wear_limit_um: 300\n", encoding="utf-8")
        assert file_digest(a) == file_digest(b)

    def test_one_changed_character_changes_the_digest(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("wear_limit_um: 300\n", encoding="utf-8")
        before = file_digest(path)
        path.write_text("wear_limit_um: 301\n", encoding="utf-8")
        assert file_digest(path) != before

    def test_missing_file_is_reported_not_raised(self, tmp_path):
        assert file_digest(tmp_path / "yok.yaml") == UNKNOWN

    def test_project_config_has_a_real_digest(self):
        digest = config_digest()
        assert digest != UNKNOWN
        assert len(digest) == 64


class TestRelativePath:
    def test_paths_inside_the_project_become_relative(self):
        from tcm.config import PROJECT_ROOT

        assert relative_path(PROJECT_ROOT / "config" / "default.yaml") == \
            "config/default.yaml"

    def test_paths_outside_the_project_stay_absolute(self, tmp_path):
        """Künye başka makinede de okunabilmeli; uydurma göreli yol üretilmez."""
        result = relative_path(tmp_path / "dışarıda.txt")
        assert "dışarıda.txt" in result


class TestRunStamp:
    def test_carries_everything_needed_to_reproduce_a_number(self):
        stamp = run_stamp()
        for key in ("git_hash", "git_dirty", "config_digest", "command",
                    "timestamp", "python", "versions"):
            assert key in stamp, f"künyede {key} yok"

    def test_records_the_libraries_that_change_results(self):
        """LightGBM sürümü değişirse ağaçlar değişir - künyede durmalı."""
        versions = package_versions()
        assert "lightgbm" in versions
        assert "numpy" in versions

    def test_command_can_be_overridden(self):
        stamp = run_stamp(command="scripts/train_model.py --feature-set param+time")
        assert stamp["command"].endswith("param+time")
