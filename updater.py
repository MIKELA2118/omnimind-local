"""Secure, opt-in update checks and installer verification."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_INSTALLER_BYTES = 1024 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
logger = logging.getLogger("omnimind.updater")


@dataclass(frozen=True, slots=True)
class UpdateRelease:
    version: str
    installer_url: str
    sha256: str
    signature: str
    release_notes: str

    @property
    def signed_payload(self) -> bytes:
        return f"{self.version}\n{self.installer_url}\n{self.sha256.lower()}".encode()


def _version_tuple(value: str) -> tuple[int, int, int]:
    if not _VERSION_PATTERN.fullmatch(value):
        raise ValueError("The update version must use the MAJOR.MINOR.PATCH format.")
    major, minor, patch = value.split(".")
    return int(major), int(minor), int(patch)


def _decode_public_key(value: str) -> Ed25519PublicKey:
    try:
        raw_key = base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("The configured update public key is not valid base64.") from error
    if len(raw_key) != 32:
        raise ValueError("The configured update public key must contain 32 bytes.")
    return Ed25519PublicKey.from_public_bytes(raw_key)


def _validate_https_url(value: str, label: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"The {label} must be an HTTPS URL.")


def _verify_release_signature(release: UpdateRelease, public_key: str) -> None:
    verifier = _decode_public_key(public_key)
    try:
        signature = base64.b64decode(release.signature, validate=True)
        verifier.verify(signature, release.signed_payload)
    except (ValueError, base64.binascii.Error, InvalidSignature) as error:
        raise ValueError("The update signature is invalid.") from error


def check_for_update(
    manifest_url: str,
    public_key: str,
    current_version: str,
) -> dict[str, Any] | None:
    """Fetch a small HTTPS manifest and return verified newer release metadata."""
    if not manifest_url:
        return None
    if not public_key:
        raise ValueError("The update channel has no trusted signing key configured.")
    _validate_https_url(manifest_url, "update manifest")
    _version_tuple(current_version)

    with httpx.Client(
        timeout=httpx.Timeout(connect=8.0, read=12.0, write=8.0, pool=8.0),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        response = client.get(manifest_url, headers={"Accept": "application/json"})
        response.raise_for_status()
        if len(response.content) > _MAX_MANIFEST_BYTES:
            raise ValueError("The update manifest exceeds the allowed size.")
        document = response.json()

    if not isinstance(document, dict):
        raise ValueError("The update manifest must be a JSON object.")
    try:
        release = UpdateRelease(
            version=str(document["version"]),
            installer_url=str(document["installer_url"]),
            sha256=str(document["sha256"]).lower(),
            signature=str(document["signature"]),
            release_notes=str(document.get("release_notes") or "")[:4000],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("The update manifest is missing required fields.") from error

    _validate_https_url(release.installer_url, "installer download")
    if not re.fullmatch(r"[a-f0-9]{64}", release.sha256):
        raise ValueError("The update manifest SHA-256 value is invalid.")
    release_version = _version_tuple(release.version)
    _verify_release_signature(release, public_key)
    if release_version <= _version_tuple(current_version):
        return None

    return {
        "version": release.version,
        "installer_url": release.installer_url,
        "sha256": release.sha256,
        "signature": release.signature,
        "release_notes": release.release_notes,
    }


def download_verified_installer(
    release_data: dict[str, Any],
    public_key: str,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Stream a signed installer to a private temp folder and verify its digest."""
    try:
        release = UpdateRelease(
            version=str(release_data["version"]),
            installer_url=str(release_data["installer_url"]),
            sha256=str(release_data["sha256"]).lower(),
            signature=str(release_data["signature"]),
            release_notes=str(release_data.get("release_notes") or "")[:4000],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("The update release data is incomplete.") from error

    _version_tuple(release.version)
    _validate_https_url(release.installer_url, "installer download")
    if not re.fullmatch(r"[a-f0-9]{64}", release.sha256):
        raise ValueError("The update SHA-256 value is invalid.")
    _verify_release_signature(release, public_key)

    with httpx.Client(
        timeout=httpx.Timeout(connect=15.0, read=90.0, write=30.0, pool=15.0),
        follow_redirects=True,
        trust_env=False,
    ) as client:
        with client.stream("GET", release.installer_url) as response:
            response.raise_for_status()
            _validate_https_url(str(response.url), "final installer download")
            content_length = response.headers.get("Content-Length")
            total = int(content_length) if content_length and content_length.isdigit() else None
            if total is not None and total > _MAX_INSTALLER_BYTES:
                raise ValueError("The update installer exceeds the 1 GiB limit.")
            digest = hashlib.sha256()
            directory = Path(tempfile.gettempdir()) / "OmniMind Local Updates"
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"OmniMind-Local-Setup-{release.version}.exe"
            written = 0
            try:
                with destination.open("wb") as installer:
                    for chunk in response.iter_bytes(_CHUNK_SIZE):
                        if not chunk:
                            continue
                        written += len(chunk)
                        if written > _MAX_INSTALLER_BYTES:
                            raise ValueError("The update installer exceeds the 1 GiB limit.")
                        installer.write(chunk)
                        digest.update(chunk)
                        if progress is not None:
                            progress(written, total)
            except Exception:
                destination.unlink(missing_ok=True)
                raise

    if digest.hexdigest() != release.sha256:
        destination.unlink(missing_ok=True)
        raise ValueError("The downloaded update does not match its signed SHA-256.")
    return destination
