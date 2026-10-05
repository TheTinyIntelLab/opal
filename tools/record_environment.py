"""Lock the installed dependency closure used by this research workflow."""
from pathlib import Path
import sys
import importlib.metadata as metadata
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from opal.artifacts import environment, write_json


def main():
    pending = ["numpy", "scipy", "torch", "scikit-learn", "matplotlib", "PyYAML", "pytest", "anndata", "h5py", "pyarrow", "pandas", "threadpoolctl"]
    versions = {}
    while pending:
        name = pending.pop()
        distribution = metadata.distribution(name)
        canonical = distribution.metadata["Name"]
        if canonical.lower() in versions:
            continue
        versions[canonical.lower()] = distribution.version
        for value in distribution.requires or []:
            requirement = Requirement(value)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    write_json(ROOT / "results/environment.json", {**environment(), "installed_research_dependency_closure": versions})
    lines = ["# Recorded Windows CPU environment, Python 3.14.0.", "# Local +cpu PyTorch wheels may require https://download.pytorch.org/whl/cpu."]
    lines += [f"{name}=={version}" for name, version in sorted(versions.items())]
    (ROOT / "requirements-lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(len(versions), "research dependencies recorded")


if __name__ == "__main__":
    main()
