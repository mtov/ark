from decimal import Decimal


def calculate_discount(subtotal):
    if subtotal >= Decimal("100.00"):
        return subtotal * Decimal("0.10")
    return Decimal("0.00")
