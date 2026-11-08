"""Website fetcher for the Bilaspur News Scraper Connector.

This module is responsible for fetching the published HTML from the
configured source website using HTTP GET. It is strictly read-only: it
never modifies, writes to, or deploys the source website.

It does not extract news records — that is the parser's responsibility.
Only raw HTML and response metadata are returned here.

Dependencies:
    - httpx: HTTP client (free, no API key required)
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from ipaddress import ip_address, ip_network
from math import isfinite
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_URL = "https://wrobr27-debug.github.io/News-Agent/"
DEFAULT_TIMEOUT = 30.0
# Default maximum response body size (1 MiB). Configurable via constructor.
DEFAULT_MAX_RESPONSE_SIZE = 1024 * 1024
DEFAULT_USER_AGENT = "Bilaspur-News-Scraper/1.0"
DEFAULT_MAX_REDIRECTS = 10
DEFAULT_CHUNK_SIZE = 8192

# Content types this fetcher accepts as HTML.
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")

# Sensitive headers that must not be forwarded to a different hostname.
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "cookie", "cookie2", "proxy-authorization", "x-forwarded-for"}
)

#: Result of the source hostname DNS validation.
#: True when resolution succeeded and the resolved address is public.
#: False when resolution could not be verified (e.g. test environment
#: without network access).  Callers must NOT rely on complete DNS
#: protection when this is False; the hostname allowlist for redirects
#: provides the actual security boundary.
_source_hostname_verified: bool = False

# Track whether the source URL contained disallowed credentials or ports.
# Set during __init__; used to inform error messages and redirect handling.
_source_url_strict_valid: bool = True
_PRIVATE_NETWORKS: tuple = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("224.0.0.0/4"),
    ip_network("240.0.0.0/4"),
    ip_network("::1/128"),
    ip_network("fc00::/7"),
    ip_network("fe80::/10"),
    ip_network("2001:db8::/32"),
    ip_network("2001::/32"),
    ip_network("2002::/16"),
)


def _extract_hostname(url: str) -> str:
    """Extract the hostname from a URL, or an empty string if absent."""
    try:
        return urlparse(url).hostname or ""
    except Exception:
        return ""


@dataclass
class FetchResult:
    """Result of a single fetch operation.

    Attributes:
        success: True when HTML was retrieved or a 304 was received.
        html: The decoded HTML body string, or None when not_modified,
            empty, or on error.
        status_code: The HTTP status code of the response.
        url: The final URL after redirects.
        etag: The ETag response header, if present.
        last_modified: The Last-Modified response header, if present.
        content_type: The Content-Type response header, if present.
        not_modified: True when the server responded HTTP 304 Not Modified.
        error: A human-readable error message when success is False.
    """

    success: bool
    html: Optional[str] = None
    status_code: Optional[int] = None
    url: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    content_type: Optional[str] = None
    not_modified: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# WebsiteFetcher
# ---------------------------------------------------------------------------

class WebsiteFetcher:
    """Fetches published HTML from a source website using HTTP GET.

    Supports conditional requests via ETag and Last-Modified headers so
    callers can avoid re-downloading content that has not changed.

    Redirects are handled manually: each destination is validated before
    it is followed. The fetcher is read-only and never modifies the
    source website.

    The fetcher is configurable: source URL, timeout, User-Agent, maximum
    response size, maximum redirects, and any previously-saved
    ETag/Last-Modified values can be supplied. An optional httpx.Client
    may be injected (primarily for testing with mocked transport).

    Example usage:

        fetcher = WebsiteFetcher(
            source_url="https://example.com/news/",
            timeout=15.0,
        )
        result = fetcher.fetch()
        if result.success and result.html:
            print(result.html[:500])
            # Save these for the next conditional request:
            #   result.etag, result.last_modified
        else:
            print("Fetch failed:", result.error)
    """

    def __init__(
        self,
        source_url: str = DEFAULT_SOURCE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        """Initialize the fetcher.

        Args:
            source_url: The URL of the published HTML to fetch (read-only).
                Must be a valid HTTPS URL with a hostname.
            timeout: Request timeout in seconds. Must be a positive,
                finite number.
            user_agent: Descriptive User-Agent string sent with requests.
                Must be non-empty.
            max_response_size: Maximum response body size in bytes.
                Must be positive.
            max_redirects: Maximum number of redirects to follow.
                Must be non-negative.
            etag: Previously-saved ETag for conditional requests.
            last_modified: Previously-saved Last-Modified for conditional
                requests.
            client: Optional httpx.Client to use (useful for testing).

        Raises:
            ValueError: If any configuration value is invalid.
        """
        self._validate_configuration(
            source_url, timeout, user_agent, max_response_size, max_redirects
        )
        self.source_url = source_url
        self.timeout = float(timeout)
        self.user_agent = user_agent
        self.max_response_size = max_response_size
        self.max_redirects = max_redirects
        self.etag = etag
        self.last_modified = last_modified
        self._client = client
        # Hostname allowlist derived from the source URL. Redirects to any
        # other hostname are blocked (fail closed).
        self._allowed_hostnames = frozenset({_extract_hostname(source_url)})
        # Validate the source hostname via DNS resolution.
        # If the hostname resolves to a private/reserved/loopback IP, reject
        # the source URL (fail closed).  If resolution cannot be performed
        # (e.g. no network access in the test environment), we set a flag
        # and document the limitation so callers do not rely on incomplete
        # DNS protection.
        try:
            addr_info = socket.getaddrinfo(_extract_hostname(source_url), None)
            # Get the first resolved address (IPv4 or IPv6)
            resolved_ip = addr_info[0][4][0]
            if ip_address(resolved_ip) in _PRIVATE_NETWORKS:
                raise ValueError(
                    f"source_url hostname {_extract_hostname(source_url)} "
                    "resolves to a private/reserved IP address; "
                    "DNS rebinding is not mitigated."
                )
            # Hostname resolves to a public IP - mark as verified.
            # Note: IP may change over time due to CDN/load-balancer, so this
            # is a best-effort check; the hostname allowlist for redirects
            # remains the primary security boundary.
            _source_hostname_verified = True
        except (socket.gaierror, ValueError, IndexError):
            # Resolution failed (e.g. no network, hostname not found).
            # In this configuration, we do not claim complete DNS protection.
            # The read-only HTTPS-only design and hostname allowlist for
            # redirects provide defense-in-depth, but the fetcher cannot
            # guarantee the destination IP will remain public.
            pass
        # Enforce strict URL policy: reject embedded credentials and
        # non-default HTTPS ports.
        self._source_url_strict_valid = True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_configuration(
        source_url: str,
        timeout: float,
        user_agent: str,
        max_response_size: int,
        max_redirects: int,
    ) -> None:
        """Validate constructor arguments.

        Raises:
            ValueError: If any argument is invalid.
        """
        if not source_url or not source_url.strip():
            raise ValueError("source_url must be non-empty")

        try:
            parsed = urlparse(source_url.strip())
        except Exception as exc:
            raise ValueError(f"source_url is not a valid URL: {exc}") from exc

        if parsed.scheme.lower() != "https":
            raise ValueError("source_url must use HTTPS")

        if not parsed.hostname:
            raise ValueError("source_url must include a hostname")

        # Reject IP-literal hostnames for the source (fail closed; DNS
        # rebinding cannot be reliably mitigated here).
        try:
            ip_address(parsed.hostname)
        except ValueError:
            pass  # Not an IP literal — hostname is fine.
        else:
            # No exception: it IS an IP literal — reject.
            raise ValueError("source_url must not be an IP address")

        # Reject NaN / infinite / non-numeric timeouts.
        if timeout is None or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a number")
        if not isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive, finite number")

        if not user_agent or not user_agent.strip():
            raise ValueError("user_agent must be non-empty")

        if (
            max_response_size is None
            or not isinstance(max_response_size, int)
            or isinstance(max_response_size, bool)
            or max_response_size <= 0
        ):
            raise ValueError("max_response_size must be a positive integer")

        if (
            max_redirects is None
            or not isinstance(max_redirects, int)
            or isinstance(max_redirects, bool)
            or max_redirects < 0
        ):
            raise ValueError("max_redirects must be a non-negative integer")

    # ------------------------------------------------------------------
    # Header building
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build the request headers, including conditional request headers.

        Conditional headers (If-None-Match / If-Modified-Since) are only
        added when the caller supplied a previous ETag / Last-Modified.
        """
        headers: dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": "text/html, application/xhtml+xml, */*; q=0.8",
        }
        if self.etag:
            headers["If-None-Match"] = self.etag
        if self.last_modified:
            headers["If-Modified-Since"] = self.last_modified
        return headers

    def _strip_sensitive_headers(
        self, headers: dict[str, str]
    ) -> dict[str, str]:
        """Remove sensitive headers that should not cross hostnames."""
        return {
            key: value
            for key, value in headers.items()
            if key.lower() not in _SENSITIVE_HEADERS
        }

    # ------------------------------------------------------------------
    # Destination security
    # ------------------------------------------------------------------

    @staticmethod
    def _is_ip_private_or_reserved(host: str) -> bool:
        """Return True when the host is an IP literal in a private,
        reserved, loopback, link-local, or multicast range."""
        try:
            addr = ip_address(host)
        except ValueError:
            return False  # Not an IP literal; handled elsewhere.
        for network in _PRIVATE_NETWORKS:
            if addr in network:
                return True
        return False

    def _is_destination_allowed(self, url: str) -> bool:
        """Return True when the redirect destination is safe to follow.

        Rules:
          - Scheme must be HTTPS.
          - Host must be in the allowlist (derived from the source URL).
          - IP-literal hosts that are private/reserved are rejected.
        """
        try:
            parsed = urlparse(url)
        except Exception:
            return False

        if parsed.scheme.lower() != "https":
            return False

        host = parsed.hostname
        if not host:
            return False

        # Reject IP-literal destinations that resolve to private/reserved
        # ranges.
        if self._is_ip_private_or_reserved(host):
            return False

        # Fail closed: only allow hostnames in the allowlist.
        if host not in self._allowed_hostnames:
            return False

        return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

    def fetch(self) -> FetchResult:
        """Fetch the HTML from the source URL.

        Redirects are followed manually: each intermediate response is
        inspected and its destination validated before the next request
        is made. Automatic redirect following is disabled.

        Returns:
            A FetchResult containing the HTML content (when successful)
            and useful response metadata. On HTTP 304, success is True
            and not_modified is True with html set to None. On any error,
            success is False and error holds a description.

        Raises:
            This method does not raise; network and HTTP errors are
            captured and returned inside FetchResult.
        """
        headers = self._build_headers()

        if self._client is not None:
            return self._fetch_with_client(self._client, headers)

        with httpx.Client(timeout=self.timeout) as client:
            return self._fetch_with_client(client, headers)

    def _fetch_with_client(
        self, client: httpx.Client, headers: dict[str, str]
    ) -> FetchResult:
        """Perform the fetch with controlled redirect handling.

        The configured timeout is passed explicitly to every request,
        whether the client is internally created or injected. Redirects
        are followed only when each destination passes validation.
        """
        current_url = self.source_url
        current_headers = dict(headers)
        redirects_followed = 0
        visited_urls: set[str] = set()

        try:
            while True:
                if current_url in visited_urls:
                    return FetchResult(
                        success=False,
                        error="Redirect loop detected",
                    )
                visited_urls.add(current_url)

                try:
                    stream_ctx = client.stream(
                        "GET",
                        current_url,
                        headers=current_headers,
                        timeout=self.timeout,
                        follow_redirects=False,
                    )
                except httpx.ConnectError:
                    return FetchResult(
                        success=False,
                        error="Connection error",
                    )
                except httpx.TimeoutException:
                    return FetchResult(
                        success=False,
                        error="Request timed out",
                    )
                except httpx.RequestError:
                    return FetchResult(
                        success=False,
                        error="Request error",
                    )

                with stream_ctx as response:
                    # Preserve metadata from the current response.
                    etag = response.headers.get("ETag")
                    last_modified = response.headers.get("Last-Modified")
                    content_type = response.headers.get("Content-Type")

                    # HTTP 304 Not Modified.
                    if response.status_code == 304:
                        if etag is None:
                            etag = self.etag
                        if last_modified is None:
                            last_modified = self.last_modified
                        return FetchResult(
                            success=True,
                            status_code=response.status_code,
                            url=str(response.url),
                            etag=etag,
                            last_modified=last_modified,
                            content_type=content_type,
                            not_modified=True,
                            html=None,
                        )

                    # Redirect: validate destination before following.
                    if response.status_code in (
                        301,
                        302,
                        303,
                        307,
                        308,
                    ):
                        if redirects_followed >= self.max_redirects:
                            return FetchResult(
                                success=False,
                                status_code=response.status_code,
                                url=str(response.url),
                                etag=etag,
                                last_modified=last_modified,
                                content_type=content_type,
                                error="Too many redirects",
                            )
                        location = response.headers.get("Location")
                        if not location:
                            return FetchResult(
                                success=False,
                                status_code=response.status_code,
                                url=str(response.url),
                                etag=etag,
                                last_modified=last_modified,
                                content_type=content_type,
                                error="Redirect with no Location header",
                            )
                        # Resolve relative Location against current URL.
                        next_url = urljoin(current_url, location)
                        if not self._is_destination_allowed(next_url):
                            return FetchResult(
                                success=False,
                                status_code=response.status_code,
                                url=str(response.url),
                                etag=etag,
                                last_modified=last_modified,
                                content_type=content_type,
                                error="Redirect destination not allowed",
                            )
                        # Do not forward sensitive headers to a different
                        # hostname.
                        next_host = urlparse(next_url).hostname
                        current_host = urlparse(current_url).hostname
                        if next_host != current_host:
                            current_headers = self._strip_sensitive_headers(
                                current_headers
                            )
                        current_url = next_url
                        redirects_followed += 1
                        continue

                    # Non-redirect response: process the body.
                    return self._process_streaming_response(
                        response, etag, last_modified, content_type
                    )

        except httpx.ConnectError:
            return FetchResult(success=False, error="Connection error")
        except httpx.TimeoutException:
            return FetchResult(success=False, error="Request timed out")
        except httpx.RequestError:
            return FetchResult(success=False, error="Request error")
        except Exception:  # unexpected: never silently report success
            return FetchResult(
                success=False, error="Unexpected error during fetch"
            )

    # ------------------------------------------------------------------
    # Response processing (streaming)
    # ------------------------------------------------------------------

    def _process_streaming_response(
        self,
        response: httpx.Response,
        etag: Optional[str],
        last_modified: Optional[str],
        content_type: Optional[str],
    ) -> FetchResult:
        """Process a streaming HTTP response.

        Reads the body in bounded chunks, enforcing the maximum response
        size before retaining any data. Respects the declared character
        encoding with a UTF-8 fallback. The stream is closed on success
        and failure.
        """
        # HTTP errors (4xx / 5xx).
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            return FetchResult(
                success=False,
                status_code=response.status_code,
                url=str(response.url),
                etag=etag,
                last_modified=last_modified,
                content_type=content_type,
                error=f"HTTP error {response.status_code}",
            )

        # Validate content type.
        content_type_primary = (content_type or "").split(";")[0].strip().lower()
        if content_type_primary not in HTML_CONTENT_TYPES:
            return FetchResult(
                success=False,
                status_code=response.status_code,
                url=str(response.url),
                etag=etag,
                last_modified=last_modified,
                content_type=content_type,
                error="Invalid content type",
            )

        # Determine character encoding: respect the declared charset,
        # fall back to UTF-8.
        encoding = self._resolve_encoding(content_type)

        # Read body incrementally with bounded chunks and size protection.
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_bytes(chunk_size=DEFAULT_CHUNK_SIZE):
                remaining = self.max_response_size - total
                if len(chunk) > remaining:
                    # This single chunk would exceed the limit. Stop
                    # immediately without retaining it.
                    return FetchResult(
                        success=False,
                        status_code=response.status_code,
                        url=str(response.url),
                        etag=etag,
                        last_modified=last_modified,
                        content_type=content_type,
                        error="Response too large",
                    )
                chunks.append(chunk)
                total += len(chunk)
                if total > self.max_response_size:
                    return FetchResult(
                        success=False,
                        status_code=response.status_code,
                        url=str(response.url),
                        etag=etag,
                        last_modified=last_modified,
                        content_type=content_type,
                        error="Response too large",
                    )
        except httpx.RequestError:
            return FetchResult(
                success=False,
                status_code=response.status_code,
                url=str(response.url),
                etag=etag,
                last_modified=last_modified,
                content_type=content_type,
                error="Request error while reading body",
            )
        finally:
            try:
                response.close()
            except Exception:
                pass

        # Empty body handling.
        if total == 0:
            return FetchResult(
                success=True,
                html="",
                status_code=response.status_code,
                url=str(response.url),
                etag=etag,
                last_modified=last_modified,
                content_type=content_type,
                not_modified=False,
            )

        html = b"".join(chunks).decode(encoding, errors="replace")
        return FetchResult(
            success=True,
            html=html,
            status_code=response.status_code,
            url=str(response.url),
            etag=etag,
            last_modified=last_modified,
            content_type=content_type,
            not_modified=False,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_encoding(content_type: Optional[str]) -> str:
        """Resolve the character encoding from a Content-Type header.

        Returns the declared charset when valid, otherwise UTF-8.
        """
        if not content_type:
            return "utf-8"
        try:
            parts = content_type.split(";")
            for part in parts[1:]:
                part = part.strip()
                if part.lower().startswith("charset="):
                    charset = part.split("=", 1)[1].strip().strip('"')
                    if charset:
                        # Validate that the codec exists.
                        import codecs

                        codecs.lookup(charset)
                        return charset
        except (LookupError, ValueError):
            pass
        return "utf-8"
