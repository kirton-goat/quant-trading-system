from official_event_sources import _extract_dated_links, _timestamp_to_iso


def test_extract_dated_links_keeps_date_with_original_link():
    html = '''<ul><li><a href="/policy/a.html">政策 A</a><span>2026-08-30</span></li>
    <li><a href="https://example.gov.cn/b.html">政策 B</a><span>2026年8月29日</span></li></ul>'''
    rows = _extract_dated_links(html, "https://example.gov.cn/list/", 10)
    assert rows == [
        ("政策 A", "2026-08-30", "https://example.gov.cn/policy/a.html"),
        ("政策 B", "2026-08-29", "https://example.gov.cn/b.html"),
    ]


def test_cninfo_timestamp_is_converted_to_china_time():
    assert _timestamp_to_iso(0) == "1970-01-01 08:00:00"


def test_extract_dated_links_handles_ndrc_nested_list_content():
    html = '''<ul><li><a href="./2026/a.html" title="政策 A">政策 A</a>
    <div class="popbox"><div>说明</div></div><span>2026/08/30</span></li></ul>'''
    rows = _extract_dated_links(html, "https://www.ndrc.gov.cn/xxgk/zcfb/", 10)
    assert rows == [("政策 A", "2026-08-30", "https://www.ndrc.gov.cn/xxgk/zcfb/2026/a.html")]
