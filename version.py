import datetime
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION_FILE = os.path.join(SCRIPT_DIR, "version.json")


def get_version():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return json.load(f)
    return {"version": "0.0.0", "date": None}


def bump_version(part="patch"):
    v = get_version()
    parts = v["version"].split(".")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1

    new_version = f"{major}.{minor}.{patch}"
    v.update(
        {
            "version": new_version,
            "date": datetime.datetime.now().isoformat(),
            "accuracy": 0.8101,
            "auc_roc": 0.8088,
        }
    )

    with open(VERSION_FILE, "w") as f:
        json.dump(v, f, indent=2)

    print(f"Version bumped to: {new_version}")
    return v


def show_version():
    v = get_version()
    print(f"Current version: {v['version']} ({v.get('date', 'N/A')})")
    return v


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        bump_version(sys.argv[1])
    else:
        show_version()
