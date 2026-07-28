def summarize_items(items):
    subtotal = sum(
        (item["unit_price"] * item["quantity"] for item in items),
        start=0,
    )
    return {
        "item_count": sum(item["quantity"] for item in items),
        "subtotal": subtotal,
    }
