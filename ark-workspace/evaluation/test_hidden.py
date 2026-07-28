from products import count_pages, paginate_products


def test_empty_catalog_has_no_pages():
    assert count_pages([], 10) == 0


def test_single_product():
    assert count_pages(["p1"], 1) == 1


def test_page_after_last_is_empty():
    assert paginate_products(list(range(20)), 3, 10) == []


def test_page_size_equal_to_catalog_size():
    products = list(range(7))
    assert count_pages(products, 7) == 1
    assert paginate_products(products, 1, 7) == products
