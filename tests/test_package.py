# -*- coding: utf-8 -*-
"""What `import kudio_enhance` gives you, and what it costs."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import kudio_enhance

SRC = Path(__file__).resolve().parent.parent / "src"


def _fresh_python(code: str, cwd: Path) -> str:
    """Run *code* in a new interpreter, where nothing has been imported yet."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SRC), env.get("PYTHONPATH")]))
    result = subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_importing_the_package_does_not_import_torch(tmp_path):
    """Configs, features and the dataset tooling are useful on a machine
    with no deep-learning framework at all, and cheap to import on one that
    has it."""
    out = _fresh_python(
        "import sys, kudio_enhance\n"
        "print('torch' in sys.modules)", cwd=tmp_path)
    assert out.strip() == "False"


def test_the_commands_that_need_no_model_do_not_import_torch(tmp_path):
    out = _fresh_python(
        "import sys, kudio_enhance\n"
        "from kudio_enhance.cli import main\n"
        "main(['models'])\n"
        "main(['init', 'config.yaml'])\n"
        "kudio_enhance.Enhancer\n"
        "print('torch loaded:', 'torch' in sys.modules)", cwd=tmp_path)
    assert "torch loaded: False" in out
    assert (tmp_path / "config.yaml").is_file()


def test_the_enhancer_is_reachable_from_the_package():
    """KudioStudio shows ``kudio_enhance.Enhancer.load('runs/exp1')`` as the
    code behind its learned-denoising panel; that has to be true."""
    from kudio_enhance.inference import Enhancer

    assert kudio_enhance.Enhancer is Enhancer
    assert "Enhancer" in kudio_enhance.__all__
    from kudio_enhance import Enhancer as imported
    assert imported is Enhancer


def test_a_misspelt_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="Enhancr"):
        kudio_enhance.Enhancr

