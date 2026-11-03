"""Verified HTTPS for hosted LLM APIs, with a separate bundled CA fallback."""

import os
import ssl
import urllib.error
import urllib.parse
import urllib.request


def _verification_error(error):
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    return reason if isinstance(reason, ssl.SSLCertVerificationError) else None


def _bundled_ca_context():
    import certifi

    # Use an independent store so an expired system root cannot shadow the
    # bundled trust chain. Hostname checks and certificate verification stay on.
    return ssl.create_default_context(cafile=certifi.where())


def urlopen_verified(request, timeout):
    """Retry only a failed TLS verification, before HTTP data can be sent.

    Explicit CA configuration remains authoritative. HTTP responses (including
    rate limits and authorization failures), timeouts and other network errors
    are never retried by this helper.
    """
    url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
    if urllib.parse.urlsplit(url).scheme.lower() != "https":
        return urllib.request.urlopen(request, timeout=timeout)
    try:
        return urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context())
    except (urllib.error.URLError, ssl.SSLError) as error:
        verification = _verification_error(error)
        if verification is None:
            raise
        original = error

    custom_ca = bool(os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"))
    fallback_error = None
    if not custom_ca:
        try:
            context = _bundled_ca_context()
        except (ImportError, OSError, ssl.SSLError) as error:
            fallback_error = str(error)
        else:
            try:
                return urllib.request.urlopen(request, timeout=timeout, context=context)
            except (urllib.error.URLError, ssl.SSLError) as error:
                if _verification_error(error) is None:
                    raise
                fallback_error = str(_verification_error(error))

    hostname = urllib.parse.urlsplit(url).hostname or "the API server"
    reason = getattr(verification, "verify_message", None) or str(verification)
    details = (
        "The configured SSL_CERT_FILE / SSL_CERT_DIR was kept; check its CA certificates."
        if custom_ca else
        "System and bundled CA verification could not establish a valid connection. "
        "Update certifi in the Python environment running ComfyUI."
    )
    if fallback_error:
        details += f" Bundled CA result: {fallback_error}."
    raise urllib.error.URLError(
        f"TLS certificate validation failed for {hostname}: {reason}. {details} "
        "Check the computer's date/time and any VPN, proxy or antivirus HTTPS inspection. "
        "An expired server/proxy certificate must be renewed. Certificate verification remains enabled."
    ) from original
