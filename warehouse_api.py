"""The warehouse team's system: stock by product and size or colour, and items held for exchanges.

Written by the Assignment 9 notebook: the lecture notebook's warehouse_api.py, unchanged. It reads where its database is from
the setting WAREHOUSE_DB."""
import os
import sqlite3
from datetime import date

DB = os.environ.get("WAREHOUSE_DB", "warehouse.db")


def _db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


_SKIP = {"the", "and", "for", "from", "with", "order", "my", "new", "old", "pair", "set", "please"}


def match_name(text, names):
    """The names that `text` could mean: every word of it appears in the name ('hiking socks' -> 'Wool hiking socks').
    When several fit, the one that ends with the last word wins ('tent' -> 'Ridgeline 2 tent', not 'Tent footprint')."""
    text = " ".join(str(text).lower().replace("-", " ").split())
    exact = [n for n in names if n.lower() == text]
    if exact:
        return exact
    stem = lambda w: w[:-1] if w.endswith("s") and len(w) > 3 else w
    words = [stem(w) for w in text.split() if len(w) > 2 and w not in _SKIP]
    hits = [n for n in names if words and all(w in n.lower() for w in words)]
    if len(hits) > 1:
        hits = [n for n in hits if stem(n.lower().split()[-1]) == words[-1]] or hits
    return hits


def _product(name):
    """The catalogue name for what the caller typed, or None if no single product fits."""
    with _db() as con:
        known = [r["product"] for r in con.execute("SELECT DISTINCT product FROM stock")]
    hits = match_name(name, known)
    return hits[0] if len(hits) == 1 else None


def _variant(product, variant):
    """The catalogue size or colour for what the caller typed: 'size L', 'large', 'the green one', 'two-burner model'."""
    want = str(variant).lower().replace("extra large", "xl").replace("extra-large", "xl")
    want = want.replace("small", "s").replace("medium", "m").replace("large", "l")
    tokens = want.replace(",", " ").replace(".", " ").split()
    with _db() as con:
        variants = [r["variant"] for r in con.execute("SELECT variant FROM stock WHERE product = ?", (product,))]
    for v in variants:
        if v.lower() == want.strip() or v.lower() in tokens:
            return v
    return None


def _available(con, product, variant):
    on_hand = con.execute("SELECT on_hand FROM stock WHERE product = ? AND variant = ?", (product, variant)).fetchone()["on_hand"]
    held = con.execute("SELECT COUNT(*) FROM reservations WHERE product = ? AND variant = ? AND status IN ('held', 'committed')",
                       (product, variant)).fetchone()[0]
    return on_hand - held


def check_stock(product: str, variant: str) -> str:
    """Check how many of ONE product, in ONE size or colour, are in stock and free to hold.
    product: the product name, for example Ridgeline 2 tent. variant: the size or colour, for example green or L."""
    p = _product(product)
    if p is None:
        return f"UNKNOWN: no single product matches {product!r}."
    v = _variant(p, variant)
    if v is None:
        with _db() as con:
            options = ", ".join(r["variant"] for r in con.execute("SELECT variant FROM stock WHERE product = ?", (p,)))
        return f"UNKNOWN: {p} does not come in {variant!r}. It comes in: {options}."
    with _db() as con:
        n = _available(con, p, v)
    return f"{p}, {v}: {n} in stock" + ("" if n > 0 else " (OUT OF STOCK)")


def hold_item(product: str, variant: str, order_id: str, request_key: str = "") -> str:
    """Hold ONE replacement item for an exchange, so that nobody else can buy it. This CHANGES the warehouse's stock.
    Call it only for an exchange that the customer asked for. It returns the hold number (RES-...).
    product: the product, for example Ridgeline 2 tent. variant: the NEW size or colour the customer wants.
    order_id: the order the exchange is for.
    request_key: optional; the same key always gets the same hold, so a repeated request never holds two items."""
    with _db() as con:
        if request_key:     # only a hold that still stands: once released, the same key holds again (the attempt was abandoned)
            old = con.execute("SELECT * FROM reservations WHERE request_key = ? AND status IN ('held', 'committed')",
                              (request_key,)).fetchone()
            if old:
                return f"HELD: {old['reservation_id']} (already held under this request key)."
    p = _product(product)
    if p is None:
        return f"REFUSED: no single product matches {product!r}."
    v = _variant(p, variant)
    if v is None:
        return f"REFUSED: {p} does not come in {variant!r}."
    with _db() as con:
        if _available(con, p, v) < 1:
            return f"REFUSED: {p}, {v} is out of stock. Nothing was held."
        res = f"RES-{5001 + con.execute('SELECT COUNT(*) FROM reservations').fetchone()[0]}"
        con.execute("INSERT INTO reservations VALUES (?, ?, ?, 1, ?, NULL, 'held', ?, ?)",
                    (res, p, v, str(order_id).strip().upper(), request_key or None, date.today().isoformat()))
    return f"HELD: {res}, one {p} ({v}) for order {str(order_id).strip().upper()}."


def release_hold(reservation_id: str) -> str:
    """Release a hold, so that the item can be sold again. Use it when an exchange will not go ahead.
    reservation_id: the hold number, for example RES-5001."""
    rid = str(reservation_id).strip().upper()
    with _db() as con:
        row = con.execute("SELECT * FROM reservations WHERE reservation_id = ?", (rid,)).fetchone()
        if row is None or row["status"] != "held":
            return f"REFUSED: there is no active hold {rid}."
        con.execute("UPDATE reservations SET status = 'released' WHERE reservation_id = ?", (rid,))
    return f"RELEASED: {rid}."


def commit_hold(reservation_id: str) -> str:
    """For the support system's code, not for a model: the exchange went ahead, so the held item ships."""
    rid = str(reservation_id).strip().upper()
    with _db() as con:
        row = con.execute("SELECT * FROM reservations WHERE reservation_id = ?", (rid,)).fetchone()
        if row is None or row["status"] != "held":
            return f"REFUSED: there is no active hold {rid}."
        con.execute("UPDATE reservations SET status = 'committed' WHERE reservation_id = ?", (rid,))
    return f"COMMITTED: {rid}."


def hold_info(reservation_id: str):
    with _db() as con:
        row = con.execute("SELECT * FROM reservations WHERE reservation_id = ?", (str(reservation_id).strip().upper(),)).fetchone()
    return dict(row) if row else None


def holds():
    with _db() as con:
        return [dict(r) for r in con.execute("SELECT * FROM reservations ORDER BY reservation_id")]
