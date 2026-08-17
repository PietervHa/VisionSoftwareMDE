"""
Password Hashing

Passwords are hashed with hashlib.scrypt -- stdlib, no extra dependency.
Each stored hash is self-describing (it embeds the cost parameters and the
salt it was created with), so the scrypt cost can be raised later without
invalidating hashes that already exist in the `users` table.

Stored format:  scrypt$n$r$p$salt_hex$hash_hex
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

_ALGORITHM = "scrypt"
_N = 2 ** 14  # CPU/memory cost factor
_R = 8        # block size
_P = 1        # parallelization factor
_KEY_LEN = 32
_SALT_LEN = 16


def hash_password(password: str) -> str:
    """Hash a plaintext password into a self-describing string, ready to store in `users.password_hash`."""
    salt = secrets.token_bytes(_SALT_LEN)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=_KEY_LEN
    )
    return f"{_ALGORITHM}${_N}${_R}${_P}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Check a plaintext password against a hash produced by hash_password().
    Returns False on any malformed/unrecognized hash rather than raising,
    so a corrupt DB row fails closed instead of crashing the login route.
    """
    try:
        algorithm, n, r, p, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != _ALGORITHM:
            return False
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, AttributeError):
        return False

    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected)
    )
    return hmac.compare_digest(derived, expected)


# A hash of a password nobody will ever type, computed once at import time.
# The login route runs verify_password() against this whenever the
# submitted username doesn't exist, so an unknown-username attempt takes
# the same amount of time as a wrong-password attempt for a real user --
# otherwise response timing alone would leak which usernames are registered.
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))
