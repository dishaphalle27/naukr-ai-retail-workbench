"""The README must not drift from the code: every command it shows has to exist."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def code_blocks():
    return "\n".join(re.findall(r"```(?:bash|text)?\n(.*?)```", README, flags=re.S))


def test_every_module_command_in_the_readme_exists():
    modules = set(re.findall(r"python -m (app(?:\.\w+)+)", code_blocks()))
    assert len(modules) >= 8
    for m in modules:
        assert importlib.util.find_spec(m) is not None, f"README mentions {m} but it does not exist"


def test_every_script_and_data_path_in_the_readme_exists():
    block = code_blocks()
    for path in set(re.findall(r"python (scripts/\w+\.py)", block)):
        assert (ROOT / path).is_file(), path
    for path in ("cases.json", "eval/cases.json", "eval/cases_edge.json", ".env.example", "requirements.txt",
                 "data/sample/planted_issues.json", "data/sample_truth/orders.csv"):
        assert (ROOT / path).exists(), f"missing {path}"


def test_readme_covers_what_the_assessment_asks_for():
    needed = ["Quick start", "Architecture", "AI-versus-code", "Data that leaves the machine", "Security", "trade-offs",
              "Known limitations", "gemini-3.5-flash", "Track: AI Engineering", "python -m app.evaluate"]
    for text in needed:
        assert text.lower() in README.lower(), f"README is missing: {text}"


def test_env_example_has_no_real_secret_and_gitignore_protects_env():
    env = (ROOT / ".env.example").read_text()
    assert re.search(r"^GEMINI_API_KEY=\s*$", env, flags=re.M)
    assert "AIza" not in env
    assert re.search(r"^\.env\s*$", (ROOT / ".gitignore").read_text(), flags=re.M)


def test_no_secret_looking_strings_anywhere_in_the_source():
    pattern = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9]{20,}")
    for path in list(ROOT.glob("app/**/*.py")) + list(ROOT.glob("tests/*.py")) + list(ROOT.glob("scripts/*.py")) + [ROOT / "README.md"]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path
