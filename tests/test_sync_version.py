import sync_version as sv


def make_repo(tmp_path, pyproject="1.2.3", badge="1.2.3", changelog="1.2.3"):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{pyproject}"\n')
    (tmp_path / "README.md").write_text(
        f"![v](https://img.shields.io/badge/version-{badge}-blue)\n"
    )
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [{changelog}] - 2026-11-05\n\n### Added\n- thing\n\n## [0.1.0]\n- old\n"
    )
    return tmp_path


def test_consistent_repo_has_no_problems(tmp_path):
    assert sv.problems(make_repo(tmp_path)) == []


def test_stale_badge_is_reported_then_fixed(tmp_path):
    root = make_repo(tmp_path, badge="1.0.0")
    assert any("README badge" in p for p in sv.problems(root))
    sv.write_badge(root)
    assert sv.problems(root) == []


def test_missing_changelog_section_is_reported(tmp_path):
    root = make_repo(tmp_path, changelog="9.9.9")
    assert any("CHANGELOG" in p for p in sv.problems(root))


def test_notes_return_only_the_requested_section(tmp_path):
    root = make_repo(tmp_path)
    notes = sv.changelog_section("1.2.3", root)
    assert "- thing" in notes
    assert "old" not in notes


def test_the_real_repository_is_consistent():
    assert sv.problems() == []
