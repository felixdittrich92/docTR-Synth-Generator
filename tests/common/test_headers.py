"""Test for python files copyright headers."""

import re
from pathlib import Path

# Matches "# Copyright (C) 2021-<YYYY>, Felix Dittrich." for any end year, so the
# check does not spuriously fail at the turn of the year while still requiring a
# correctly formatted copyright line plus the exact license body.
COPYRIGHT_RE = re.compile(r"^# Copyright \(C\) 2021-\d{4}, Felix Dittrich\.\n")
LICENSE_BODY = (
    "\n# This program is licensed under the Apache License 2.0.\n"
    "# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.\n"
)


def test_copyright_header():
    excluded_files = ["__init__.py", "version.py"]
    invalid_files = []
    locations = [".github", "generator"]

    for location in locations:
        for source_path in Path(__file__).parent.parent.parent.joinpath(location).rglob("*.py"):
            if source_path.name not in excluded_files:
                content = source_path.read_text()
                if not (COPYRIGHT_RE.match(content) and LICENSE_BODY in content):
                    invalid_files.append(source_path)
    assert len(invalid_files) == 0, f"Invalid copyright header in the following files: {invalid_files}"
