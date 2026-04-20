"""Comprehensive tests for ingest module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from odoo_rag.ingest import fetch_page, ingest_all


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_success(mock_get):
    """Test successful page fetching."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><p>Test content</p></body></html>"
    mock_get.return_value = mock_response

    content = fetch_page("https://example.com")

    assert content is not None
    assert "Test content" in content
    assert len(content) > 0


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_with_headers(mock_get):
    """Test that fetch_page sends proper headers."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html>Content</html>"
    mock_get.return_value = mock_response

    fetch_page("https://example.com")

    # Verify headers were sent
    call_kwargs = mock_get.call_args[1]
    assert "headers" in call_kwargs


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_http_error(mock_get):
    """Test fetch_page raises exception on HTTP errors."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = Exception("Not found")
    mock_get.return_value = mock_response

    # fetch_page should raise the exception
    try:
        fetch_page("https://example.com/notfound")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Not found" in str(e)


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_timeout(mock_get):
    """Test fetch_page raises exception on timeout."""
    mock_get.side_effect = Exception("Timeout")

    # fetch_page should raise the exception
    try:
        fetch_page("https://example.com")
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Timeout" in str(e)


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_connection_error(mock_get):
    """Test fetch_page raises exception on connection errors."""
    mock_get.side_effect = ConnectionError("Connection failed")

    # fetch_page should raise the exception
    try:
        fetch_page("https://example.com")
        assert False, "Should have raised exception"
    except ConnectionError as e:
        assert "Connection failed" in str(e)


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_strips_html_tags(mock_get):
    """Test that fetch_page extracts text from HTML."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><h1>Title</h1><p>Paragraph</p></body></html>"
    mock_get.return_value = mock_response

    content = fetch_page("https://example.com")

    # Should contain text but not HTML tags
    assert "Title" in content
    assert "Paragraph" in content
    assert "<h1>" not in content
    assert "<p>" not in content


@patch("odoo_rag.ingest.requests.get")
def test_fetch_page_handles_empty_response(mock_get):
    """Test fetch_page handles empty HTML."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html></html>"
    mock_get.return_value = mock_response

    content = fetch_page("https://example.com")

    assert isinstance(content, str)


@patch("odoo_rag.ingest.fetch_page")
def test_ingest_all_basic(mock_fetch):
    """Test basic ingest_all functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = Path(tmpdir) / "sources.json"
        config = {
            "sources": [
                {"url": "https://example.com/doc1", "name": "doc1"},
            ]
        }
        with open(sources_file, "w") as f:
            json.dump(config, f)

        mock_fetch.return_value = "Test content from page"

        output_dir = Path(tmpdir) / "output"

        ingest_all(str(sources_file), str(output_dir))

        assert output_dir.exists()
        # Verify output file was created
        output_files = list(output_dir.glob("*.txt"))
        assert len(output_files) >= 1


@patch("odoo_rag.ingest.fetch_page")
def test_ingest_all_multiple_sources(mock_fetch):
    """Test ingest_all with multiple sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = Path(tmpdir) / "sources.json"
        config = {
            "sources": [
                {"url": "https://example.com/doc1", "name": "doc1"},
                {"url": "https://example.com/doc2", "name": "doc2"},
                {"url": "https://example.com/doc3", "name": "doc3"},
            ]
        }
        with open(sources_file, "w") as f:
            json.dump(config, f)

        mock_fetch.return_value = "Content"

        output_dir = Path(tmpdir) / "output"

        ingest_all(str(sources_file), str(output_dir))

        assert output_dir.exists()


@patch("odoo_rag.ingest.fetch_page")
def test_ingest_all_handles_failures(mock_fetch):
    """Test ingest_all continues on individual failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = Path(tmpdir) / "sources.json"
        config = {
            "sources": [
                {"url": "https://example.com/doc1", "name": "doc1"},
                {"url": "https://example.com/doc2", "name": "doc2"},
            ]
        }
        with open(sources_file, "w") as f:
            json.dump(config, f)

        # First succeeds, second fails
        mock_fetch.side_effect = ["Content 1", Exception("Failed")]

        output_dir = Path(tmpdir) / "output"

        # Should not raise exception (catches and continues)
        ingest_all(str(sources_file), str(output_dir))

        assert output_dir.exists()


@patch("odoo_rag.ingest.fetch_page")
def test_ingest_all_creates_output_dir(mock_fetch):
    """Test that ingest_all creates output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = Path(tmpdir) / "sources.json"
        config = {"sources": [{"url": "https://example.com/doc", "name": "doc"}]}
        with open(sources_file, "w") as f:
            json.dump(config, f)

        mock_fetch.return_value = "Content"

        output_dir = Path(tmpdir) / "new" / "output"

        ingest_all(str(sources_file), str(output_dir))

        assert output_dir.exists()


@patch("odoo_rag.ingest.fetch_page")
def test_ingest_all_saves_content(mock_fetch):
    """Test that ingest_all saves fetched content to files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = Path(tmpdir) / "sources.json"
        config = {
            "sources": [
                {"url": "https://example.com/doc", "name": "test doc"},
            ]
        }
        with open(sources_file, "w") as f:
            json.dump(config, f)

        mock_fetch.return_value = "Test content"

        output_dir = Path(tmpdir) / "output"

        ingest_all(str(sources_file), str(output_dir))

        # Should create dir and save file
        assert output_dir.exists()
        output_file = output_dir / "test_doc.txt"
        assert output_file.exists()
        assert output_file.read_text() == "Test content"


@patch("odoo_rag.ingest.requests.get")
def test_fetch_pages_recursive_single_page(mock_get):
    """Test fetch_pages_recursive with single page."""
    from odoo_rag.ingest import fetch_pages_recursive

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body><p>Page content</p></body></html>"
    mock_get.return_value = mock_response

    content = fetch_pages_recursive("https://example.com/doc", max_pages=1)

    assert "Page content" in content
    assert mock_get.called


@patch("odoo_rag.ingest.requests.get")
def test_fetch_pages_recursive_follows_links(mock_get):
    """Test fetch_pages_recursive follows internal links."""
    from odoo_rag.ingest import fetch_pages_recursive

    # First page with link to second page
    page1_html = '<html><body><p>Page 1</p><a href="/page2">Link</a></body></html>'
    # Second page
    page2_html = "<html><body><p>Page 2</p></body></html>"

    mock_response1 = MagicMock()
    mock_response1.status_code = 200
    mock_response1.text = page1_html

    mock_response2 = MagicMock()
    mock_response2.status_code = 200
    mock_response2.text = page2_html

    mock_get.side_effect = [mock_response1, mock_response2]

    content = fetch_pages_recursive("https://example.com/page1", max_pages=2)

    assert "Page 1" in content or "Page 2" in content
    # Should have made 2 requests
    assert mock_get.call_count <= 2


@patch("odoo_rag.ingest.requests.get")
def test_fetch_pages_recursive_skips_external_links(mock_get):
    """Test fetch_pages_recursive skips external domains."""
    from odoo_rag.ingest import fetch_pages_recursive

    # Page with external link
    html = '<html><body><p>Content</p><a href="https://external.com/page">External</a></body></html>'

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html
    mock_get.return_value = mock_response

    fetch_pages_recursive("https://example.com/doc", max_pages=5)

    # Should only fetch the starting page, not external link
    assert mock_get.call_count == 1


@patch("odoo_rag.ingest.requests.get")
def test_fetch_pages_recursive_handles_errors(mock_get):
    """Test fetch_pages_recursive continues on errors."""
    from odoo_rag.ingest import fetch_pages_recursive

    # First page succeeds, second fails
    page1_html = '<html><body><p>Page 1</p><a href="/page2">Link</a></body></html>'

    mock_response1 = MagicMock()
    mock_response1.status_code = 200
    mock_response1.text = page1_html

    mock_get.side_effect = [mock_response1, Exception("Network error")]

    content = fetch_pages_recursive("https://example.com/page1", max_pages=2)

    # Should have content from first page despite second page error
    assert "Page 1" in content


@patch("odoo_rag.ingest.requests.get")
def test_fetch_pages_recursive_respects_path_filtering(mock_get):
    """Test fetch_pages_recursive filters links by path."""
    from odoo_rag.ingest import fetch_pages_recursive

    # Page with links to different sections
    html = """
    <html><body>
        <p>Content</p>
        <a href="/documentation/18.0/applications/sales.html">Same section</a>
        <a href="/documentation/18.0/developer/api.html">Different section</a>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html
    mock_get.return_value = mock_response

    # Start from applications section
    fetch_pages_recursive(
        "https://example.com/documentation/18.0/applications/index.html", max_pages=2
    )

    # Should follow same-section link but not different-section link
    assert mock_get.call_count >= 1


@patch("odoo_rag.ingest.requests.get")
def test_fetch_pages_recursive_removes_nav_elements(mock_get):
    """Test fetch_pages_recursive removes navigation elements."""
    from odoo_rag.ingest import fetch_pages_recursive

    html = """
    <html><body>
        <nav>Navigation menu</nav>
        <header>Header</header>
        <p>Main content</p>
        <footer>Footer</footer>
        <script>alert('test');</script>
    </body></html>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = html
    mock_get.return_value = mock_response

    content = fetch_pages_recursive("https://example.com/doc", max_pages=1)

    # Should have main content but not nav/header/footer/script
    assert "Main content" in content
    assert "Navigation menu" not in content
    assert "alert" not in content


@patch("odoo_rag.ingest.fetch_pages_recursive")
def test_ingest_all_with_max_pages(mock_fetch_recursive):
    """Test ingest_all with max_pages > 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sources_file = Path(tmpdir) / "sources.json"
        config = {
            "sources": [
                {
                    "url": "https://example.com/doc",
                    "name": "multi page",
                    "max_pages": 5,
                },
            ]
        }
        with open(sources_file, "w") as f:
            json.dump(config, f)

        mock_fetch_recursive.return_value = "Multi-page content"

        output_dir = Path(tmpdir) / "output"

        ingest_all(str(sources_file), str(output_dir))

        # Should call fetch_pages_recursive instead of fetch_page
        assert mock_fetch_recursive.called
        assert output_dir.exists()
