from app.preprocess import is_url_like_text


def test_is_url_like_text_detects_https_blob() -> None:
    text = "httpswwwinquirercomnewsphiladelphiapolicecarvideowestunrestchildbackseat20201028html"
    assert is_url_like_text(text) is True


def test_is_url_like_text_detects_www_blob() -> None:
    text = "httpswwwwnycstudiosorgpodcastsradiolabarticlesdispatches1918"
    assert is_url_like_text(text) is True


def test_is_url_like_text_returns_false_for_normal_sentence() -> None:
    text = "get all software instantly are you tired of 30 days shipping delays"
    assert is_url_like_text(text) is False


def test_is_url_like_text_returns_false_for_empty_string() -> None:
    assert is_url_like_text("") is False


def test_is_url_like_text_returns_false_for_whitespace_only() -> None:
    assert is_url_like_text("   ") is False


def test_is_url_like_text_returns_false_for_short_non_url_token() -> None:
    assert is_url_like_text("software123") is False


def test_is_url_like_text_returns_false_for_text_with_spaces() -> None:
    text = "httpswwwexamplecom this should not be treated as one url blob"
    assert is_url_like_text(text) is False


def test_is_url_like_text_detects_long_web_like_blob_without_http_prefix() -> None:
    text = "myaccountverifyloginexamplecomsecurehtml"
    assert is_url_like_text(text) is True