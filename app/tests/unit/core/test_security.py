"""Tests for security utilities"""

from datetime import timedelta

import pytest


class TestPasswordHashing:
    """Test password hashing and verification"""

    def test_verify_password_correct(self):
        """Test verifying a correct password"""
        # Arrange
        from app.core.security import hash_password, verify_password

        plain_password = "securepassword123"
        hashed_password = hash_password(plain_password)

        # Act
        is_valid = verify_password(plain_password, hashed_password)

        # Assert
        assert is_valid is True

    def test_verify_password_incorrect(self):
        """Test verifying an incorrect password"""
        # Arrange
        from app.core.security import hash_password, verify_password

        plain_password = "wrongpassword"
        hashed_password = hash_password("securepassword123")

        # Act
        is_valid = verify_password(plain_password, hashed_password)

        # Assert
        assert is_valid is False

    def test_hash_password(self):
        """Test hashing a password"""
        # Arrange
        from app.core.security import hash_password, verify_password

        plain_password = "mypassword123"

        # Act
        hashed = hash_password(plain_password)

        # Assert
        assert hashed is not None
        assert hashed != plain_password
        assert hashed.startswith("$2b$")  # bcrypt hash prefix
        assert verify_password(plain_password, hashed) is True

    def test_hash_password_different_hashes(self):
        """Test that hashing the same password twice produces different hashes"""
        # Arrange
        from app.core.security import hash_password

        password = "samepassword"

        # Act
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # Assert
        assert hash1 != hash2  # Different due to salt
        # But both should verify correctly
        from app.core.security import verify_password

        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True

    def test_hash_password_empty(self):
        """Test hashing an empty password"""
        # Arrange
        from app.core.security import hash_password

        password = ""

        # Act
        hashed = hash_password(password)

        # Assert
        assert hashed is not None
        assert hashed.startswith("$2b$")


class TestJWTToken:
    """Test JWT token creation and verification"""

    def test_create_access_token(self):
        """Test creating an access token"""
        # Arrange
        from app.core.security import create_access_token

        data = {"sub": "user@example.com"}

        # Act
        token = create_access_token(data)

        # Assert
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_expiration(self):
        """Test creating an access token with custom expiration"""
        # Arrange
        from app.core.security import create_access_token

        data = {"sub": "user@example.com"}
        expires_delta = timedelta(minutes=30)

        # Act
        token = create_access_token(data, expires_delta)

        # Assert
        assert token is not None

    def test_decode_access_token_valid(self):
        """Test decoding a valid access token"""
        # Arrange
        from app.core.security import create_access_token, decode_access_token

        data = {"sub": "user@example.com", "user_id": 123}
        token = create_access_token(data)

        # Act
        decoded = decode_access_token(token)

        # Assert
        assert decoded is not None
        assert decoded["sub"] == "user@example.com"
        assert decoded["user_id"] == 123
        assert "exp" in decoded

    def test_decode_access_token_invalid(self):
        """Test decoding an invalid access token"""
        # Arrange
        from app.core.security import decode_access_token

        invalid_token = "invalid.token.string"

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            decode_access_token(invalid_token)
        assert "token" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_decode_access_token_expired(self):
        """Test decoding an expired access token"""
        # Arrange
        from app.core.security import create_access_token, decode_access_token

        data = {"sub": "user@example.com"}
        # Create token that's already expired
        expired_delta = timedelta(seconds=-1)
        token = create_access_token(data, expired_delta)

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            decode_access_token(token)
        assert "expired" in str(exc_info.value).lower() or "token" in str(exc_info.value).lower()

    def test_create_refresh_token(self):
        """Test creating a refresh token"""
        # Arrange
        from app.core.security import create_refresh_token

        data = {"sub": "user@example.com"}

        # Act
        token = create_refresh_token(data)

        # Assert
        assert token is not None
        assert isinstance(token, str)


class TestPasswordValidation:
    """Test password validation utilities"""

    def test_validate_password_strong(self):
        """Test validating a strong password"""
        # Arrange
        from app.core.security import validate_password

        password = "StrongP@ssw0rd123"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_password_too_short(self):
        """Test validating a password that's too short"""
        # Arrange
        from app.core.security import validate_password

        password = "Short1!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("at least 8 characters" in err.lower() for err in result["errors"])

    def test_validate_password_no_uppercase(self):
        """Test validating a password without uppercase letters"""
        # Arrange
        from app.core.security import validate_password

        password = "lowercase123!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("uppercase" in err.lower() for err in result["errors"])

    def test_validate_password_no_lowercase(self):
        """Test validating a password without lowercase letters"""
        # Arrange
        from app.core.security import validate_password

        password = "UPPERCASE123!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("lowercase" in err.lower() for err in result["errors"])

    def test_validate_password_no_digit(self):
        """Test validating a password without digits"""
        # Arrange
        from app.core.security import validate_password

        password = "NoDigits!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("digit" in err.lower() for err in result["errors"])

    def test_validate_password_multiple_errors(self):
        """Test validating a password with multiple errors"""
        # Arrange
        from app.core.security import validate_password

        password = "short"  # Too short, no uppercase, no digit, no special char

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 2


class TestJWTLibraryContract:
    """Pins from the python-jose → PyJWT migration (2026-10-01).

    python-jose drags the `ecdsa` package, which carries an unfixable
    timing-attack CVE (PYSEC-2026-1325 / CVE-2024-23342 — no fixed
    release upstream). The swap must be invisible on the wire: same
    HS256 tokens, same claims, tokens minted by pre-migration deploys
    still verify.
    """

    # Minted with python-jose before it left the dependency tree
    # (HS256 is RFC-identical between the libraries, so the static
    # token keeps proving compat in CI without jose installed).
    LEGACY_TOKEN = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiJsZWdhY3ktdXNlckBleGFtcGxlLmNvbSIsInVzZXJfaWQiOjQyLC"
        "J0eXBlIjoiYWNjZXNzIiwiZXhwIjoyMTA2MjMwNzI2fQ."
        "tUy1EEKf2pwbej0X2Yp24t30UWFI_Q9zQRs6_j0fDjo"
    )
    LEGACY_KEY = "legacy-compat-test-key-0123456789abcdef"

    def test_legacy_jose_encoded_token_still_verifies(self):
        """Tokens issued by pre-migration deployments (python-jose)
        must verify under PyJWT — live sessions must not be invalidated
        by the library swap."""
        # Arrange
        from unittest.mock import patch

        from app.config.settings import settings
        from app.core.security import decode_access_token

        with patch.object(settings, "SECRET_KEY", self.LEGACY_KEY):
            # Act
            decoded = decode_access_token(self.LEGACY_TOKEN)

        # Assert
        assert decoded["sub"] == "legacy-user@example.com"
        assert decoded["user_id"] == 42

    def test_wire_exp_claim_is_integer(self):
        """The exp claim must be an integer on the wire. Neither jose
        nor PyJWT may leak a datetime/float through — a datetime exp is
        exactly the kind of input a future PyJWT release could stop
        accepting, so the encoder pins the wire format itself."""
        # Arrange
        import base64
        import json

        from app.core.security import create_access_token

        token = create_access_token({"sub": "wire-format@example.com"})

        # Act
        payload_b64 = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))

        # Assert
        assert isinstance(claims["exp"], int)

    def test_decode_failures_raise_invalid_token_error(self):
        """Public contract: every decode failure raises PyJWT's
        InvalidTokenError (supertype of ExpiredSignatureError,
        DecodeError, ...)."""
        # Arrange
        from jwt.exceptions import InvalidTokenError

        from app.core.security import create_access_token, decode_access_token

        expired = create_access_token({"sub": "x"}, timedelta(seconds=-60))

        # Act & Assert
        with pytest.raises(InvalidTokenError):
            decode_access_token("not-a-jwt")
        with pytest.raises(InvalidTokenError):
            decode_access_token(expired)

    def test_alg_none_token_rejected(self):
        """A forged alg=none token must never decode — the verifier
        pins the algorithm instead of trusting the header."""
        # Arrange
        import base64
        import json

        from jwt.exceptions import InvalidTokenError

        from app.core.security import decode_access_token

        def _b64(obj: dict[str, object]) -> str:
            raw = json.dumps(obj).encode()
            return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

        forged = (
            _b64({"alg": "none", "typ": "JWT"})
            + "."
            + _b64({"sub": "attacker", "exp": 4102444800})
            + "."
        )

        # Act & Assert
        with pytest.raises(InvalidTokenError):
            decode_access_token(forged)
