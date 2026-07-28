from decimal import Decimal
from orders import create_order
from reports import summarize_items


ITEMS = [
    {"unit_price": Decimal("30.00"), "quantity": 2},
    {"unit_price": Decimal("20.00"), "quantity": 1},
]


def test_order_subtotal():
    assert create_order(ITEMS)["subtotal"] == Decimal("80.00")


def test_report_subtotal():
    assert summarize_items(ITEMS)["subtotal"] == Decimal("80.00")
