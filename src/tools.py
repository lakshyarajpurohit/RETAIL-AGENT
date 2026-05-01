import pandas as pd
from langchain.tools import tool
from src.data_loader import RetailDataManager
from datetime import datetime

data_manager = RetailDataManager()

CURRENT_DATE = datetime(2026, 5, 1)


@tool
def search_products(query: str, size: str = None, max_price: float = None, prefer_sale: bool = False) -> str:
    """
    Search the product catalog by style, occasion, or tag keywords (e.g. 'modest evening lace').
    Optionally filter by size (as a string like '8' or '12'), max price, and whether to prefer sale items.
    Always use this tool when the customer describes what they are looking for.
    Returns top 5 matching products sorted by sale status and bestseller score.
    """
    df = data_manager.get_inventory().copy()

    # ── Price filter ──────────────────────────────────────────
    if max_price is not None:
        df = df[df['price'] <= float(max_price)]

    # ── Size filter (FIX: size stored as string keys, not int) ─
    if size is not None:
        size_str = str(size).strip()
        df = df[df['stock_per_size'].apply(
            lambda x: isinstance(x, dict) and x.get(size_str, 0) > 0
        )]

    # ── Sale preference ───────────────────────────────────────
    if prefer_sale:
        df = df[df['is_sale'] == True]

    # ── Keyword search across tags + title ───────────────────
    keywords = [kw.strip() for kw in query.lower().split() if len(kw.strip()) > 2]
    if keywords:
        mask = pd.Series([False] * len(df), index=df.index)
        for kw in keywords:
            mask = (
                mask
                | df['title'].str.lower().str.contains(kw, na=False)
                | df['tags'].str.lower().str.contains(kw, na=False)
                | df['vendor'].str.lower().str.contains(kw, na=False)
            )
        df = df[mask]

    if df.empty:
        return (
            "No products matched all the specified filters. "
            "Try relaxing the size, price, or style criteria."
        )

    # ── Sort: sale first, then bestseller_score desc ──────────
    df = df.sort_values(by=['is_sale', 'bestseller_score'], ascending=[False, False])
    top = df.head(5)

    lines = ["Here are the best matching products:\n"]
    for _, row in top.iterrows():
        size_stock = f"Stock for size {size}: {row['stock_per_size'].get(str(size), 0)}" if size else ""
        sale_info = "[ON SALE]" if row['is_sale'] else ("[CLEARANCE]" if row['is_clearance'] else "")
        lines.append(
            f"• {row['product_id']} — {row['title']} {sale_info}\n"
            f"  Vendor: {row['vendor']} | Price: ${row['price']} (was ${row['compare_at_price']})\n"
            f"  Tags: {row['tags']} | Bestseller Score: {row['bestseller_score']}/100\n"
            f"  {size_stock}\n"
        )

    return "\n".join(lines)


@tool
def get_product(product_id: str) -> str:
    """
    Fetch complete details for a specific product using its Product ID (e.g. P0042).
    Use this when the customer mentions a specific product code starting with 'P'.
    """
    clean_id = str(product_id).strip().upper()
    product = data_manager.get_product_by_id(clean_id)

    if product is None:
        return f"Error: Product '{clean_id}' does not exist in the catalog."

    stock_summary = ", ".join(
        [f"Size {k}: {v}" for k, v in product['stock_per_size'].items()]
    ) if isinstance(product['stock_per_size'], dict) else str(product['stock_per_size'])

    return (
        f"Product ID: {product['product_id']}\n"
        f"Name: {product['title']}\n"
        f"Vendor: {product['vendor']}\n"
        f"Price: ${product['price']} (Original: ${product['compare_at_price']})\n"
        f"Tags: {product['tags']}\n"
        f"On Sale: {'Yes' if product['is_sale'] else 'No'} | "
        f"Clearance: {'Yes' if product['is_clearance'] else 'No'}\n"
        f"Bestseller Score: {product['bestseller_score']}/100\n"
        f"Stock by Size: {stock_summary}"
    )


@tool
def get_order(order_id: str) -> str:
    """
    Retrieve order details (date, product, size, amount paid) for a given Order ID (e.g. O0005).
    Use this when the customer asks about a specific order or mentions an order number.
    """
    clean_id = str(order_id).strip().upper()
    if not clean_id.startswith("O"):
        clean_id = f"O{clean_id}"

    order = data_manager.get_order_by_id(clean_id)

    if order is None:
        return (
            f"Error: Order '{clean_id}' was not found in the system. "
            "Please double-check the order ID and try again."
        )

    return (
        f"Order ID: {order['order_id']}\n"
        f"Order Date: {order['order_date']}\n"
        f"Product ID: {order['product_id']}\n"
        f"Size Purchased: {order['size']}\n"
        f"Amount Paid: ${order['price_paid']}\n"
        f"Customer ID: {order['customer_id']}"
    )


@tool
def evaluate_return(order_id: str) -> str:
    """
    Evaluate whether an order is eligible for a return or exchange.
    Applies the full return policy: clearance rules, sale rules, vendor exceptions (Aurelia Couture, Nocturne).
    Always use this tool when a customer asks about returning or exchanging an item.
    Never guess — always call this tool.
    """
    clean_id = str(order_id).strip().upper()
    if not clean_id.startswith("O"):
        clean_id = f"O{clean_id}"

    # ── Step 1: Fetch order ───────────────────────────────────
    order = data_manager.get_order_by_id(clean_id)
    if order is None:
        return (
            f"Decision: DENIED.\n"
            f"Reason: Order '{clean_id}' was not found in the system. "
            "Cannot process a return for an order that does not exist."
        )

    # ── Step 2: Fetch product ─────────────────────────────────
    product = data_manager.get_product_by_id(order['product_id'])
    if product is None:
        return (
            f"Decision: DENIED.\n"
            f"Reason: The product associated with order '{clean_id}' could not be found."
        )

    # ── Step 3: Calculate days since order ────────────────────
    order_date = datetime.strptime(str(order['order_date']), '%Y-%m-%d')
    days_since = (CURRENT_DATE - order_date).days

    vendor = str(product['vendor'])
    is_clearance = bool(product['is_clearance'])
    is_sale = bool(product['is_sale'])

    # ── Step 4: Apply policy rules in strict priority order ───

    # Rule 1: Clearance → absolute final sale
    if is_clearance:
        return (
            f"Decision: DENIED — Final Sale Item.\n"
            f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
            f"Reason: This is a CLEARANCE item. Clearance items are final sale and are not eligible "
            f"for return or exchange under any circumstances."
        )

    # Rule 2: Aurelia Couture → exchange only (no refund), 14-day window
    if vendor == 'Aurelia Couture':
        if days_since <= 14:
            return (
                f"Decision: APPROVED — Exchange Only.\n"
                f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
                f"Reason: Aurelia Couture items are eligible for EXCHANGE ONLY (no cash refund). "
                f"The order is within the 14-day exchange window ({14 - days_since} days remaining)."
            )
        else:
            return (
                f"Decision: DENIED — Exchange Window Expired.\n"
                f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
                f"Reason: Aurelia Couture items allow exchanges only within 14 days. "
                f"The window expired {days_since - 14} days ago."
            )

    # Rule 3: Nocturne → 21-day extended window (FIX: checked BEFORE sale rule)
    if vendor == 'Nocturne':
        window = 21
        refund_type = "Store Credit" if is_sale else "Full Refund"
        if days_since <= window:
            return (
                f"Decision: APPROVED — {refund_type}.\n"
                f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
                f"Reason: Nocturne has an extended 21-day return window. "
                f"This order is within the window ({window - days_since} days remaining). "
                f"Refund type: {refund_type}."
            )
        else:
            return (
                f"Decision: DENIED — Return Window Expired.\n"
                f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
                f"Reason: Nocturne's extended 21-day return window has expired by {days_since - window} days."
            )

    # Rule 4: Sale item → 7-day window, store credit only
    if is_sale:
        window = 7
        if days_since <= window:
            return (
                f"Decision: APPROVED — Store Credit Only.\n"
                f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
                f"Reason: This is a sale item. Returns are accepted within 7 days for store credit only. "
                f"This order qualifies ({window - days_since} days remaining)."
            )
        else:
            return (
                f"Decision: DENIED — Return Window Expired.\n"
                f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
                f"Reason: Sale items have a 7-day return window. "
                f"This window expired {days_since - window} days ago."
            )

    # Rule 5: Normal item → 14-day window, full refund
    window = 14
    if days_since <= window:
        return (
            f"Decision: APPROVED — Full Refund.\n"
            f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
            f"Reason: Standard item within the 14-day return window ({window - days_since} days remaining). "
            f"Customer is eligible for a full refund."
        )
    else:
        return (
            f"Decision: DENIED — Return Window Expired.\n"
            f"Order: {clean_id} | Product: {product['title']} | Ordered: {order['order_date']} ({days_since} days ago)\n"
            f"Reason: The standard 14-day return window has expired by {days_since - window} days."
        )