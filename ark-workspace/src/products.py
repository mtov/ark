def count_pages(products, page_size):
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    return (len(products) + page_size - 1) // page_size


def paginate_products(products, page, page_size):
    if page <= 0:
        raise ValueError("page must be positive")
    total_pages = count_pages(products, page_size)
    if page > total_pages:
        return []
    start = (page - 1) * page_size
    end = start + page_size
    return products[start:end]
