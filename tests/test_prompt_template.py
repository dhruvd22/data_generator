import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nl_sql_generator.prompt_builder import load_template_messages


def test_load_template_messages(tmp_path):
    tmpl_name = "_tmp_prompt.txt"
    tmpl_dir = os.path.join(os.path.dirname(__file__), "..", "nl_sql_generator", "prompt_template")
    os.makedirs(tmpl_dir, exist_ok=True)
    path = os.path.join(tmpl_dir, tmpl_name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("### role: system\nfoo\n### role: user\n{{nl_question}}")
    try:
        msgs = load_template_messages(tmpl_name, {"t": 1}, "Q?")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "Q?" in msgs[1]["content"]
    finally:
        os.remove(path)
