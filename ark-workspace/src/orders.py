from pricing import calculate_discount


def create_order(items):
    subtotal = sum(
        (item["unit_price"] * item["quantity"] for item in items),
        start=0,
    )
    discount = calculate_discount(subtotal)
    return {
        "items": items,
        "subtotal": subtotal,
        "discount": discount,
        "total": subtotal - discount,
    }
