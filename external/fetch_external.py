"""Download the external (other-games) training data into $MTAH_EXT_DIR (default ./external_data).

- DFL/Sportec IDSSE open data, 7 matches, from the Hugging Face mirror pysport/idsse-data
  (figshare, the primary host, blocks scripted downloads).
- Metrica Sports open sample data (git clone of metrica-sports/sample-data).
"""

import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("MTAH_EXT_DIR", Path(__file__).resolve().parent.parent / "external_data"))
KIND = {"matchinformation": "meta", "events_raw": "event", "positions_raw_observed": "tracking"}


def fetch_dfl():
    from huggingface_hub import hf_hub_download, list_repo_files
    raw = ROOT / "dfl" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in list_repo_files("pysport/idsse-data", repo_type="dataset"):
        if not name.endswith(".xml"):
            continue
        match_id = name.split("DFL-MAT-")[1][:-4]
        kind = next(v for k, v in KIND.items() if k in name)
        target = raw / "{}_{}.xml".format(match_id, kind)
        if not target.exists():
            shutil.copy(hf_hub_download("pysport/idsse-data", name, repo_type="dataset"), target)
        print("dfl", match_id, kind)


def fetch_metrica():
    raw = ROOT / "metrica" / "raw"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/metrica-sports/sample-data", str(raw)])
    print("metrica sample data at", raw)


if __name__ == "__main__":
    fetch_dfl()
    fetch_metrica()
