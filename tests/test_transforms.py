from app.scanners.transforms import transformed_texts


def test_markdown_url_extraction_transform() -> None:
    transformed = list(transformed_texts("[x](https://example.test/?token=abc)"))
    assert ("markdown_url_canary_match", "https://example.test/?token=abc") in transformed
