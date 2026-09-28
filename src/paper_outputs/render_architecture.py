"""Stage the author-edited final Figure 2, without redrawing its layout.

The submitted PNG was exported by the author after manual PowerPoint edits.
The PDF wraps that same raster; it is not editable vector artwork.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def render(output: Path, reference_root: Path | None = None) -> tuple[Path, Path]:
    reference = reference_root or Path(__file__).resolve().parents[2] / "figures/reference"
    sources = [reference / f"Figure_2.{suffix}" for suffix in ("png", "pdf")]
    if not all(source.is_file() for source in sources):
        raise FileNotFoundError("Provide the checkout's figures/reference directory for the author-edited Figure 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = []
    for source in sources:
        target = output / source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        targets.append(target)
    return tuple(targets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args(argv)
    for path in render(args.output_dir, args.reference_root):
        print(path)


if __name__ == "__main__":
    main()
