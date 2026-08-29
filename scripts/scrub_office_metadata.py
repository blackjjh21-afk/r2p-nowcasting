#!/usr/bin/env python3
"""Remove author/machine metadata and normalize an OOXML container.

The public release includes an editable Figure 6 PPTX and a convenience XLSX
containing only aggregate tables.  Office containers are ZIP archives and can
retain author names, obsolete document titles, wall-clock timestamps and
custom properties even when their visible contents are safe.  This helper
scrubs those package properties and rewrites the archive deterministically.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


FIXED_W3CDTF = "2000-01-01T00:00:00Z"
FIXED_ZIP_TIME = (2000, 1, 1, 0, 0, 0)

CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
CUSTOM = "http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

for prefix, namespace in (
    ("cp", CP),
    ("dc", DC),
    ("dcterms", DCTERMS),
    ("xsi", XSI),
    ("vt", VT),
):
    ET.register_namespace(prefix, namespace)


def q(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def scrub_core_xml(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    for tag in (
        q(DC, "title"),
        q(DC, "subject"),
        q(DC, "creator"),
        q(CP, "keywords"),
        q(DC, "description"),
        q(CP, "lastModifiedBy"),
        q(CP, "category"),
        q(CP, "contentStatus"),
    ):
        element = root.find(tag)
        if element is not None:
            element.text = None

    revision = root.find(q(CP, "revision"))
    if revision is not None:
        revision.text = "1"
    for name in ("created", "modified"):
        element = root.find(q(DCTERMS, name))
        if element is None:
            element = ET.SubElement(root, q(DCTERMS, name))
        element.text = FIXED_W3CDTF
        element.set(q(XSI, "type"), "dcterms:W3CDTF")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def scrub_app_xml(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    for name in ("Company", "Manager", "HyperlinkBase"):
        element = root.find(q(EP, name))
        if element is not None:
            element.text = None
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def empty_custom_xml(payload: bytes) -> bytes:
    root = ET.fromstring(payload)
    for child in list(root):
        root.remove(child)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def scrub_office_file(path: Path) -> None:
    """Scrub and deterministically rewrite ``path`` in place."""

    path = Path(path)
    if path.suffix.lower() not in {".pptx", ".xlsx"}:
        raise ValueError(f"expected .pptx or .xlsx: {path}")
    temporary = path.with_suffix(path.suffix + ".scrubbed")
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as target:
        for name in sorted(source.namelist()):
            payload = source.read(name)
            if name == "docProps/core.xml":
                payload = scrub_core_xml(payload)
            elif name == "docProps/app.xml":
                payload = scrub_app_xml(payload)
            elif name == "docProps/custom.xml":
                payload = empty_custom_xml(payload)

            original = source.getinfo(name)
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = original.external_attr
            info.create_system = original.create_system
            target.writestr(
                info,
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.files:
        scrub_office_file(path)
        print(path)


if __name__ == "__main__":
    main()
