"""Seal and verify one authorized Open Yap copy and its location record."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile

from .open_yap_full_plan import protected_scratch
from .sources import data_root


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def encrypt(gpg_home, fingerprint, source, output):
    """Publish a ciphertext only after GPG has completed successfully."""
    with NamedTemporaryFile(dir=output.parent) as encrypted:
        status = subprocess.run(["gpg", "--homedir", str(gpg_home), "--batch", "--no-tty",
                                 "--trust-model", "always", "--recipient", fingerprint,
                                 "--encrypt", "--output", "-", "--", str(source)],
                                stdout=encrypted, stderr=subprocess.DEVNULL).returncode
        if status:
            raise ValueError("recipient encryption failed")
        encrypted.flush()
        os.fsync(encrypted.fileno())
        os.link(encrypted.name, output)


def copy_verify(source, destination):
    if subprocess.run(["rclone", "copyto", "--immutable", str(source), destination],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise ValueError("private object upload failed")
    with subprocess.Popen(["rclone", "cat", destination], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL) as process:
        actual = hashlib.file_digest(process.stdout, "sha256").hexdigest()
        if process.wait() or actual != digest(source):
            raise ValueError("private ciphertext readback failed")


def seal(root, source, gpg_home, fingerprint, object_id, expected_sha, delivery):
    if not (root / ".campaign-machine").is_file():
        raise ValueError("authorized data stays on Vast")
    source = source.resolve()
    scratch = protected_scratch(source.parent)
    gpg_home = gpg_home.resolve()
    if not gpg_home.is_relative_to(scratch):
        raise ValueError("the seal keyring must stay within protected scratch")
    if (not re.fullmatch(r"[0-9A-Fa-f]{40}", fingerprint)
            or not re.fullmatch(r"[0-9a-f]{32}", object_id)):
        raise ValueError("use a verified full key fingerprint and an opaque object ID")
    actual = digest(source)
    if actual != expected_sha:
        raise ValueError("the source object digest changed")
    store = os.environ.get("SUPERTURN_REMOTE", "").rstrip("/")
    if not re.fullmatch(r"[A-Za-z0-9_-]+:[A-Za-z0-9._-]+", store):
        raise ValueError("the configured private object store is missing")
    directory = root / "raw" / "open-yap" / object_id
    directory.mkdir(parents=True, mode=0o700)
    asset, ledger = directory / "payload.gpg", directory / "copy-location.gpg"
    remote = f"{store}/superturn/raw/open-yap/{object_id}"
    destination = f"{remote}/{asset.name}"
    encrypt(gpg_home, fingerprint, source, asset)
    record = {"delivery": delivery, "plaintext_sha256": actual, "ciphertext_sha256": digest(asset),
              "plaintext_bytes": source.stat().st_size, "ciphertext_bytes": asset.stat().st_size,
              "sealed_at": datetime.now(timezone.utc).isoformat(), "vast_ciphertext": str(asset),
              "r2_ciphertext": destination, "r2_copy_record": f"{remote}/{ledger.name}"}
    with NamedTemporaryFile(dir=scratch) as private:
        private.write(json.dumps(record, indent=2).encode())
        private.flush()
        encrypt(gpg_home, fingerprint, Path(private.name), ledger)
    # Publish the sealed custody record first. If an asset copy fails, the
    # private ledger still identifies the expected object and the Vast copy.
    copy_verify(ledger, record["r2_copy_record"])
    copy_verify(asset, destination)
    return {"verified_ciphertext_bytes": asset.stat().st_size, "copy_record_verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--gpg-home", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--delivery", required=True)
    args = parser.parse_args()
    print(json.dumps(seal(data_root(), args.source, args.gpg_home, args.fingerprint,
                          args.object_id, args.sha256, args.delivery), indent=2))
