import pytest
from unittest.mock import MagicMock, patch
from scraper.frontier import Frontier
from scraper.recursion import recursive_crawl

def test_frontier_push_pop():
    """Test basic Frontier operations (FIFO/LIFO)."""
    # BFS (Queue)
    # Use different lengths to ensure deterministic sort order (shorter = higher priority)
    f_bfs = Frontier("http://example.com", MagicMock(), same_domain_only=True, strategy="bfs")
    f_bfs.push_links("http://example.com", 1, ["http://example.com/long", "http://example.com/s"], lambda u, i: MagicMock())
    
    assert f_bfs.pop()[0] == "http://example.com/s"
    assert f_bfs.pop()[0] == "http://example.com/long"
    
    # DFS (Stack)
    f_dfs = Frontier("http://example.com", MagicMock(), same_domain_only=True, strategy="dfs")
    f_dfs.push_links("http://example.com", 1, ["http://example.com/long", "http://example.com/s"], lambda u, i: MagicMock())
    
    # DFS pops LIFO. push_links adds [s, long] (sorted by score).
    # Stack: [s, long]. Pop -> long.
    # WAIT: push_links reverses candidates for DFS.
    # candidates = [s, long] (sorted by length)
    # reversed = [long, s]
    # append long, append s.
    # Stack: [long, s]
    # Pop -> s
    assert f_dfs.pop()[0] == "http://example.com/s"
    assert f_dfs.pop()[0] == "http://example.com/long"

def test_frontier_deduplication():
    """Test that Frontier ignores duplicates."""
    f = Frontier("http://example.com", MagicMock(), same_domain_only=True)
    f.push_links("http://example.com", 1, ["http://example.com/a"], lambda u, i: MagicMock())
    f.push_links("http://example.com", 1, ["http://example.com/a"], lambda u, i: MagicMock())
    
    assert f.pop()[0] == "http://example.com/a"
    assert f.pop() is None

@patch("scraper.recursion.crawl_single_node_with_pagination")
def test_recursive_crawl_limits(mock_crawl, tmp_path):
    """Test that recursion respects max_total_pages."""
    # Mock crawl to return 1 saved page every time
    mock_crawl.return_value = (["some/path.md"], {"links": {"internal": ["http://example.com/next"]}}, "http://example.com/final")
    
    count = recursive_crawl(
        start_url="http://example.com",
        max_total_pages=2,
        max_depth=5
    )
    
    assert count == 2
    assert mock_crawl.call_count == 2
