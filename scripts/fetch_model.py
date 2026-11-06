"""Download the DLWMLS nnU-Net weights into the package tree at build time.

dlwmls_pipeline.py fetches them at runtime, into this same directory, when it
finds the model folder absent. Doing it here instead means the image works
offline and works when the container is run as a non-root user, which cannot
write /DLWMLS.

Deliberately uses only the standard library rather than huggingface_hub: the
hub client is a runtime dependency resolved afresh on every build, and a build
that merely needs to copy five files down should not break when its HTTP stack
changes. It has already done so once -- hub 2.4.0 with httpx2 2.14.0 raises
TypeError inside its own response decoder on these files.

If /run/secrets/ca.pem exists it is used as the trust store, for building on a
host behind TLS inspection.
"""

import json
import os
import shutil
import ssl
import urllib.request
from pathlib import Path

REPO = "nichart/DLWMLS"
REVISION = "main"
DEST = Path("/DLWMLS/DLWMLS")
API = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}"
RESOLVE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
CA = "/run/secrets/ca.pem"

_ctx = ssl.create_default_context(cafile=CA) if os.path.isfile(CA) else ssl.create_default_context()


def _open(url):
    request = urllib.request.Request(url, headers={"User-Agent": "nichart-dlwmls-build"})
    return urllib.request.urlopen(request, context=_ctx, timeout=300)


def _wanted(name):
    """Keep the trained model; skip what the pipeline never loads.

    checkpoint_best.pth is 246MB and unused -- dlwmls_pipeline.py asks for
    checkpoint_final.pth. The loss curve and training log are build artefacts
    of the model, not inputs to inference.
    """
    if not name.startswith("nnunet_results/"):
        return False
    base = os.path.basename(name)
    return base not in ("checkpoint_best.pth", "progress.png") and not base.startswith("training_log_")


with _open(API) as response:
    listing = json.load(response)

names = sorted(f["rfilename"] for f in listing.get("siblings", []) if _wanted(f["rfilename"]))
if not names:
    raise SystemExit(f"{REPO}@{REVISION} listed no nnunet_results/ files; has the model moved?")

for name in names:
    out = DEST / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with _open(RESOLVE + name) as response, open(out, "wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"fetched {name} ({out.stat().st_size} bytes)")
