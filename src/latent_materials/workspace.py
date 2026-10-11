"""Prepare isolated experiment workspaces from the distributed data."""

from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
import json
import os
import shutil


def resources(name):
    return json.loads(files("latent_materials").joinpath("resources", name).read_text())


def prepare(data_root, destination):
    """Copy reference inputs into a new writable workspace, preserving source data."""
    data_root, destination = Path(data_root).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new output directory: {destination}")
    layout = resources("data_layout.json")
    missing = [
        relative
        for relative in layout
        if not (data_root / Path(relative).relative_to("data")).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing reference data: {missing[:5]}")
    destination.mkdir(parents=True)
    for relative, logical in layout.items():
        target = destination / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_root / Path(relative).relative_to("data"), target)
    return destination


def check_inputs(workspace, experiment):
    """Report required inputs before an expensive experiment starts."""
    requirements = resources("required_inputs.json")[experiment]
    return [
        str(Path(workspace) / p)
        for p in requirements["files"]
        if not (Path(workspace) / p).is_file()
    ]


@contextmanager
def working_directory(path):
    """Restore the calling directory even if a numerical experiment fails."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def require_dependencies(*args, **kwargs):
    raise RuntimeError(
        "Install the research dependency extra before running this experiment."
    )


def display(*objects, **kwargs):
    """Display notebook tables when IPython is available."""
    try:
        from IPython.display import display as show
    except ImportError:
        for obj in objects:
            print(obj)
    else:
        show(*objects, **kwargs)
