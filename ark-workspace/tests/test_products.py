from products import count_pages, paginate_products


def test_count_pages_for_partial_page():
    assert count_pages(list(range(21)), 10) == 3


def test_count_pages_for_exact_multiple():
    assert count_pages(list(range(20)), 10) == 2


def test_paginate_products():
    products = list(range(12))
    assert paginate_products(products, 2, 5) == [5, 6, 7, 8, 9]
