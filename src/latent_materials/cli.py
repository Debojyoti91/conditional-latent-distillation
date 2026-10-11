"""List and run named scientific experiments."""

import argparse
from importlib import import_module
from pathlib import Path
from .workspace import resources, prepare, check_inputs, working_directory


def run_experiment(
    name, *, data_root, output, model_root=None, experiment_options=None
):
    """Run a named experiment using copied inputs and a new output directory."""
    registry = resources("experiments.json")
    if name not in registry:
        raise ValueError(f"Unknown experiment: {name}")
    layout = resources("data_layout.json")
    sources = {
        logical: Path(data_root).resolve() / Path(relative).relative_to("data")
        for relative, logical in layout.items()
    }
    required = resources("required_inputs.json")[name]["files"]
    missing = [
        logical
        for logical in required
        if not sources.get(logical, Path("/missing-input")).is_file()
        and not (model_root is not None and (Path(model_root) / logical).is_file())
    ]
    if missing:
        raise FileNotFoundError(
            "Required private or missing inputs:\n" + "\n".join(missing)
        )
    workspace = prepare(data_root, output)
    if model_root is not None:
        import shutil

        for logical in required:
            source = Path(model_root) / logical
            if source.is_file() and logical not in sources:
                target = workspace / logical
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    missing = check_inputs(workspace, name)
    if missing:
        raise FileNotFoundError(
            "Required private or missing inputs:\n" + "\n".join(missing)
        )
    import matplotlib

    matplotlib.use("Agg")
    run = import_module(registry[name]["module"]).run
    with working_directory(workspace):
        return run(workspace, **(experiment_options or {}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "run"])
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Optional private checkpoint tree; never included in the repository",
    )
    args = parser.parse_args(argv)
    registry = resources("experiments.json")
    if args.command == "list":
        for name, details in registry.items():
            print(f"{name} [{details['execution']}]: {details['description']}")
        return 0
    if not args.experiment or args.experiment not in registry:
        parser.error("Choose an experiment shown by list.")
    if args.output is None:
        parser.error("run requires --output pointing to a new directory.")
    run_experiment(
        args.experiment,
        data_root=args.data_root,
        output=args.output,
        model_root=args.model_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
