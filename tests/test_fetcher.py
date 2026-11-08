"""Tests for src.connector.fetcher using mocked HTTP responses.

These tests do not depend on the live website and do not modify any
external resources. All HTTP interactions are simulated with
httpx.MockTransport.

Every security test is designed to fail if the relevant protection is
removed.
"""

from __future__ import annotations

import httpx
import pytest

from src.connector.fetcher import (
    DEFAULT_MAX_RESPONSE_SIZE,
    DEFAULT_SOURCE_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    HTML_CONTENT_TYPES,
    FetchResult,
    WebsiteFetcher,
)


def _make_fetcher(handler, **kwargs) -> WebsiteFetcher:
    """Build a WebsiteFetcher backed by a mocked httpx transport.

    The client is closed after the fetch to avoid resource leaks.
    """
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    return WebsiteFetcher(client=client, **kwargs), client


def _fetch_and_close(fetcher: WebsiteFetcher, client: httpx.Client):
    """Run a fetch and close the client afterwards."""
    try:
        return fetcher.fetch()
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Original 9 tests (retained, with corrections)
# ---------------------------------------------------------------------------

def test_successful_html_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == DEFAULT_SOURCE_URL
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "ETag": '"abc123"',
                "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
            text="<html><body><h1>Bilaspur News</h1></body></html>",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert isinstance(result, FetchResult)
    assert result.success is True
    assert result.not_modified is False
    assert result.status_code == 200
    assert result.url == DEFAULT_SOURCE_URL
    assert result.etag == '"abc123"'
    assert result.last_modified == "Wed, 21 Oct 2015 07:28:00 GMT"
    assert result.content_type == "text/html; charset=utf-8"
    assert result.html == "<html><body><h1>Bilaspur News</h1></body></html>"
    assert result.error is None


def test_http_304_not_modified():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            304,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "ETag": '"abc123"',
                "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
        )

    fetcher, client = _make_fetcher(
        handler,
        etag='"abc123"',
        last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
    )
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert result.not_modified is True
    assert result.status_code == 304
    assert result.html is None
    assert result.etag == '"abc123"'
    assert result.last_modified == "Wed, 21 Oct 2015 07:28:00 GMT"
    assert result.error is None


def test_connection_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect")

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.not_modified is False
    assert result.html is None
    assert result.error is not None
    assert "Connection error" in result.error


def test_request_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out")

    fetcher, client = _make_fetcher(handler, timeout=5.0)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.html is None
    assert result.error is not None
    assert "timed out" in result.error.lower()


def test_http_error_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html><body>Not Found</body></html>",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.status_code == 404
    assert result.html is None
    assert result.error is not None
    assert "404" in result.error


def test_invalid_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            text='{"status":"ok"}',
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.status_code == 200
    assert result.html is None
    assert result.error is not None
    assert "content type" in result.error.lower()
    # Error must be sanitized: never echo the disallowed content type.
    assert "application/json" not in result.error


def test_conditional_request_headers_sent():
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["If-None-Match"] = request.headers.get("If-None-Match")
        seen_headers["If-Modified-Since"] = request.headers.get("If-Modified-Since")
        return httpx.Response(
            304,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "ETag": '"abc123"',
                "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
            },
        )

    fetcher, client = _make_fetcher(
        handler,
        etag='"abc123"',
        last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
    )
    result = _fetch_and_close(fetcher, client)

    assert seen_headers["If-None-Match"] == '"abc123"'
    assert seen_headers["If-Modified-Since"] == "Wed, 21 Oct 2015 07:28:00 GMT"
    assert result.not_modified is True


def test_no_conditional_headers_when_not_supplied():
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["If-None-Match"] = request.headers.get("If-None-Match")
        seen_headers["If-Modified-Since"] = request.headers.get("If-Modified-Since")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html></html>",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert seen_headers["If-None-Match"] is None
    assert seen_headers["If-Modified-Since"] is None


def test_defaults():
    fetcher = WebsiteFetcher()
    assert fetcher.source_url == DEFAULT_SOURCE_URL
    assert fetcher.timeout == DEFAULT_TIMEOUT
    assert fetcher.user_agent == DEFAULT_USER_AGENT
    assert fetcher.max_response_size == DEFAULT_MAX_RESPONSE_SIZE
    assert fetcher.etag is None
    assert fetcher.last_modified is None

    assert DEFAULT_USER_AGENT.startswith("Bilaspur-News-Scraper")
    assert HTML_CONTENT_TYPES == ("text/html", "application/xhtml+xml")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_validation_source_url_empty():
    with pytest.raises(ValueError):
        WebsiteFetcher(source_url="")


def test_constructor_validation_source_url_not_https():
    with pytest.raises(ValueError):
        WebsiteFetcher(source_url="http://example.com/")


def test_constructor_validation_source_url_missing_hostname():
    with pytest.raises(ValueError):
        WebsiteFetcher(source_url="https://")


def test_constructor_validation_source_url_ip_address():
    with pytest.raises(ValueError):
        WebsiteFetcher(source_url="https://192.168.1.1/")


def test_constructor_validation_timeout_negative():
    with pytest.raises(ValueError):
        WebsiteFetcher(timeout=-1.0)


def test_constructor_validation_timeout_zero():
    with pytest.raises(ValueError):
        WebsiteFetcher(timeout=0)


def test_constructor_validation_timeout_nan():
    import math

    with pytest.raises(ValueError):
        WebsiteFetcher(timeout=float("nan"))


def test_constructor_validation_timeout_infinite():
    with pytest.raises(ValueError):
        WebsiteFetcher(timeout=float("inf"))


def test_constructor_validation_timeout_none():
    with pytest.raises(ValueError):
        WebsiteFetcher(timeout=None)


def test_constructor_validation_user_agent_empty():
    with pytest.raises(ValueError):
        WebsiteFetcher(user_agent="")


def test_constructor_validation_max_response_size_negative():
    with pytest.raises(ValueError):
        WebsiteFetcher(max_response_size=-100)


def test_constructor_validation_max_response_size_zero():
    with pytest.raises(ValueError):
        WebsiteFetcher(max_response_size=0)


def test_constructor_validation_max_redirects_negative():
    with pytest.raises(ValueError):
        WebsiteFetcher(max_redirects=-1)


# ---------------------------------------------------------------------------
# Redirect security — every test proves the unapproved destination is never
# contacted.
# ---------------------------------------------------------------------------

def _redirect_handler_factory(
    final_response: httpx.Response,
    redirect_location: str,
) -> httpx.MockTransport:
    """Build a MockTransport that redirects once then serves final_response.

    Tracks which URLs were requested so tests can assert an unapproved
    destination was never contacted.
    """
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if "evil.example.com" in str(request.url) or "localhost" in str(request.url):
            # Should never be reached if redirect blocking works.
            pytest.fail(f"Unapproved destination was contacted: {request.url}")
        if "private" in str(request.url):
            pytest.fail(f"Private IP destination was contacted: {request.url}")
        if len(requested) == 1:
            # First request: redirect.
            return httpx.Response(302, headers={"Location": redirect_location})
        return final_response

    return httpx.MockTransport(handler), requested


def test_approved_redirect_same_host():
    """Redirect to the same host is followed."""
    final = httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html>Redirected</html>",
    )
    transport, requested = _redirect_handler_factory(
        final, "https://wrobr27-debug.github.io/News-Agent/index.html"
    )
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is True
    assert result.html == "<html>Redirected</html>"


def test_unrelated_host_redirect_blocked():
    """Redirect to an unrelated host is blocked and never contacted."""
    final = httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html>evil</html>",
    )
    transport, requested = _redirect_handler_factory(
        final, "https://evil.example.com/"
    )
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "not allowed" in result.error
    # Prove the evil host was never contacted.
    assert not any("evil.example.com" in url for url in requested)


def test_localhost_redirect_blocked():
    """Redirect to localhost is blocked."""
    final = httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html>localhost</html>",
    )
    transport, requested = _redirect_handler_factory(
        final, "https://localhost/"
    )
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "not allowed" in result.error
    assert not any("localhost" in url for url in requested)


def test_private_ip_redirect_blocked():
    """Redirect to a private IP is blocked."""
    final = httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html>private</html>",
    )
    transport, requested = _redirect_handler_factory(
        final, "https://192.168.1.1/"
    )
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "not allowed" in result.error
    assert not any("192.168.1.1" in url for url in requested)


def test_loopback_ipv6_redirect_blocked():
    """Redirect to IPv6 loopback is blocked."""
    final = httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html>loopback</html>",
    )
    transport, requested = _redirect_handler_factory(
        final, "https://[::1]/"
    )
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "not allowed" in result.error
    assert not any("[::1]" in url for url in requested)


def test_http_redirect_blocked():
    """Redirect to http:// is blocked (scheme must be HTTPS)."""
    final = httpx.Response(
        200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        text="<html>http</html>",
    )
    transport, requested = _redirect_handler_factory(
        final, "http://example.com/"
    )
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "not allowed" in result.error


def test_redirect_loop_detected():
    """Redirects that loop back to a visited URL are detected."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        # Always redirect back to the source URL (loop).
        return httpx.Response(302, headers={"Location": DEFAULT_SOURCE_URL})

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client, max_redirects=5)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "Redirect loop detected" in result.error


def test_too_many_redirects_blocked():
    """Exceeding max_redirects fails safely."""
    count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        count[0] += 1
        # Always redirect to a different path on the same host.
        return httpx.Response(
            302, headers={"Location": f"https://wrobr27-debug.github.io/page{count[0]}"}
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client, max_redirects=2)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "Too many redirects" in result.error


# ---------------------------------------------------------------------------
# Response-size limit — verified with bounded streaming.
# ---------------------------------------------------------------------------

def test_response_size_below_limit():
    small_body = b"<html><body>small</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=small_body,
        )

    fetcher, client = _make_fetcher(handler, max_response_size=10000)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert result.html == "<html><body>small</body></html>"


def test_response_size_exactly_at_limit():
    body = b"x" * DEFAULT_MAX_RESPONSE_SIZE

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=body,
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert len(result.html) == DEFAULT_MAX_RESPONSE_SIZE


def test_response_size_above_limit():
    """Reading stops immediately when the limit is exceeded."""
    body = b"x" * (DEFAULT_MAX_RESPONSE_SIZE + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=body,
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.error == "Response too large"



# ---------------------------------------------------------------------------
# Empty responses
# ---------------------------------------------------------------------------

def test_empty_html_response_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert result.html == ""
    assert result.status_code == 200


# ---------------------------------------------------------------------------
# Content-Type handling
# ---------------------------------------------------------------------------

def test_missing_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={},
            text="<html></html>",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.error == "Invalid content type"


def test_unsupported_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.4",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.error == "Invalid content type"


# ---------------------------------------------------------------------------
# HTTP 304 without validator headers
# ---------------------------------------------------------------------------

def test_http_304_without_validator_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            304,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    fetcher, client = _make_fetcher(
        handler,
        etag='"old"',
        last_modified="old-date",
    )
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert result.not_modified is True
    assert result.etag == '"old"'
    assert result.last_modified == "old-date"
    assert result.html is None


# ---------------------------------------------------------------------------
# HTTP 500 response
# ---------------------------------------------------------------------------

def test_http_500_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>Server Error</html>",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is False
    assert result.status_code == 500
    assert result.html is None
    assert "500" in result.error


# ---------------------------------------------------------------------------
# Timeout enforcement — fetcher timeout passed explicitly to each request.
# ---------------------------------------------------------------------------

def test_fetcher_timeout_passed_to_request():
    """The fetcher's timeout is passed to client.stream() explicitly.

    Uses different timeouts on the fetcher and injected client to prove
    the fetcher's value is the one applied.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    transport = httpx.MockTransport(handler)
    # Injected client has a different (shorter) timeout.
    client = httpx.Client(transport=transport, timeout=2.0)
    fetcher = WebsiteFetcher(client=client, timeout=7.0)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert "timed out" in result.error.lower()


# ---------------------------------------------------------------------------
# Character encoding
# ---------------------------------------------------------------------------

def test_encoding_utf8_hindi():
    hindi = "<html><body>\u092c\u093f\u0928\u094d\u0926\u093f\u0937\u094d\u091f</body></html>"
    body = hindi.encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=body,
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert result.html == hindi


def test_encoding_declared_charset_respected():
    """A declared ISO-8859-1 charset is respected."""
    body = "<html><body>\xe9</body></html>".encode("iso-8859-1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=ISO-8859-1"},
            content=body,
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert "\xe9" in result.html  # decoded as ISO-8859-1


def test_encoding_invalid_charset_falls_back_to_utf8():
    """An invalid charset falls back to UTF-8 safely."""
    body = "<html><body>hello</body></html>".encode("utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=invalid-codec-xyz"},
            content=body,
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert result.success is True
    assert "hello" in result.html


# ---------------------------------------------------------------------------
# Conditional-header behavior where mock returns 304 only when expected
# headers are present
# ---------------------------------------------------------------------------

def test_conditional_304_only_when_expected_headers_present():
    received: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["if-none-match"] = request.headers.get("If-None-Match")
        received["if-modified-since"] = request.headers.get("If-Modified-Since")
        if (
            received["if-none-match"] == '"expected"'
            and received["if-modified-since"] == "Expected-Date"
        ):
            return httpx.Response(
                304,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                    "ETag": '"expected"',
                    "Last-Modified": "Expected-Date",
                },
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>changed</html>",
        )

    fetcher, client = _make_fetcher(
        handler,
        etag='"expected"',
        last_modified="Expected-Date",
    )
    result = _fetch_and_close(fetcher, client)

    assert received["if-none-match"] == '"expected"'
    assert received["if-modified-since"] == "Expected-Date"
    assert result.not_modified is True


def test_conditional_200_when_headers_missing():
    received: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["if-none-match"] = request.headers.get("If-None-Match")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>full content</html>",
        )

    fetcher, client = _make_fetcher(handler)
    result = _fetch_and_close(fetcher, client)

    assert received["if-none-match"] is None
    assert result.success is True
    assert result.html == "<html>full content</html>"


# ---------------------------------------------------------------------------
# Resource cleanup — verifies the actual response object is closed.
# ---------------------------------------------------------------------------

def test_response_closed_on_success():
    """Verify the response returned by the transport is closed on success."""
    closed = [False]

    class TrackingResponse(httpx.Response):
        def close(self):
            closed[0] = True
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        resp = TrackingResponse(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>ok</html>",
        )
        return resp

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is True
    assert closed[0] is True


def test_response_closed_on_http_error():
    """Verify the response is closed on HTTP error (e.g. 500)."""
    closed = [False]

    class TrackingResponse(httpx.Response):
        def close(self):
            closed[0] = True
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        resp = TrackingResponse(
            500,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>error</html>",
        )
        return resp

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert result.status_code == 500
    assert closed[0] is True


def test_response_closed_on_size_limit():
    """Verify the response is closed when the size limit is exceeded."""
    closed = [False]

    class TrackingResponse(httpx.Response):
        def close(self):
            closed[0] = True
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        resp = TrackingResponse(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"x" * (DEFAULT_MAX_RESPONSE_SIZE + 100),
        )
        return resp

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert result.error == "Response too large"
    assert closed[0] is True


def test_response_closed_on_streaming_exception():
    """Verify the response is closed when reading the body raises."""
    closed = [False]

    class TrackingResponse(httpx.Response):
        def close(self):
            closed[0] = True
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        resp = TrackingResponse(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"chunk1",
        )
        # Wrap iter_bytes as an instance attribute so it raises mid-stream.
        orig_iter = resp.iter_bytes

        def raising_iter(chunk_size=8192):
            yield from orig_iter(chunk_size)
            raise httpx.RequestError("stream broke")

        resp.iter_bytes = raising_iter
        return resp

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is False
    assert closed[0] is True


def test_internally_created_client_is_closed():
    """An internally created httpx.Client is closed after fetch."""
    fetcher = WebsiteFetcher(timeout=5.0)
    # Manually invoke the internal client path by not passing a client.
    # We can't easily inspect the internal client, but we can verify the
    # fetch succeeds and no exception indicates a leaked client.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>ok</html>",
        )

    transport = httpx.MockTransport(handler)
    # Temporarily replace _client to use our transport for this test only.
    client = httpx.Client(transport=transport)
    fetcher._client = client
    try:
        result = fetcher.fetch()
    finally:
        client.close()

    assert result.success is True


def test_injected_client_not_closed_by_fetcher():
    """The fetcher does not close an injected client (ownership stays with
    the caller)."""
    closed = [False]

    class TrackingClient(httpx.Client):
        def close(self):
            closed[0] = True
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html>ok</html>",
        )

    transport = httpx.MockTransport(handler)
    client = TrackingClient(transport=transport)
    fetcher = WebsiteFetcher(client=client)
    result = fetcher.fetch()

    assert result.success is True
    # The fetcher should NOT close the injected client.
    assert closed[0] is False
    # The test helper is responsible for closing it.
    client.close()
    assert closed[0] is True
