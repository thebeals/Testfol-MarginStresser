import json

import pandas as pd
import pytest

from app.services import ndx_components


class FakeResponse:
    def __init__(self, *, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def _components(tickers):
    weight = 100.0 / len(tickers)
    return pd.DataFrame(
        {
            "Ticker": tickers,
            "Name": [f"Company {ticker}" for ticker in tickers],
            "Weight": weight,
        }
    )


def _snapshot(tickers, source):
    return ndx_components.ComponentSnapshot(
        components=_components(tickers),
        as_of=pd.Timestamp("2026-08-06"),
        source=source,
        weight_basis="test weights",
    )


def test_fetch_nasdaq_components_parses_date_tickers_and_proxy_weights():
    rows = [
        {
            "symbol": f"T{i:03d}" if i else "ABC.A",
            "companyName": f"Company {i} Common Stock",
            "marketCap": f"{(i + 1) * 1_000:,}",
        }
        for i in range(100)
    ]
    payload = {
        "data": {
            "date": "Aug 6, 2026",
            "data": {"rows": rows},
        }
    }
    client = FakeClient(FakeResponse(payload=payload))

    snapshot = ndx_components.fetch_nasdaq_components(client)

    assert snapshot.as_of == pd.Timestamp("2026-08-06")
    assert snapshot.source == "Nasdaq API"
    assert len(snapshot.components) == 100
    assert "ABC-A" in set(snapshot.components["Ticker"])
    assert snapshot.components["Weight"].sum() == pytest.approx(100.0)
    assert snapshot.components.iloc[0]["Ticker"] == "T099"
    assert client.calls[0][0] == ndx_components.NASDAQ_COMPONENTS_URL


def test_fetch_wikipedia_components_extracts_current_table():
    rows = "".join(
        f"<tr><td>{'ABC.A' if i == 0 else f'T{i:03d}'}</td><td>Company {i}</td></tr>"
        for i in range(100)
    )
    html = f"""
        <html><body>
        <table><tr><th>Unrelated</th></tr><tr><td>Value</td></tr></table>
        <table><tr><th>Ticker</th><th>Company</th></tr>{rows}</table>
        </body></html>
    """
    client = FakeClient(FakeResponse(text=html))

    snapshot = ndx_components.fetch_wikipedia_components(
        client,
        retrieved_at=pd.Timestamp("2026-08-07"),
    )

    assert snapshot.as_of == pd.Timestamp("2026-08-07")
    assert len(snapshot.components) == 100
    assert "ABC-A" in set(snapshot.components["Ticker"])
    assert snapshot.components["Weight"].sum() == pytest.approx(100.0)
    assert "equal-weight fallback" in snapshot.weight_basis


def test_load_current_components_marks_matching_wikipedia_verification(monkeypatch):
    tickers = [f"T{i:03d}" for i in range(100)]
    monkeypatch.setattr(
        ndx_components,
        "fetch_nasdaq_components",
        lambda _client=None: _snapshot(tickers, "Nasdaq API"),
    )
    monkeypatch.setattr(
        ndx_components,
        "fetch_wikipedia_components",
        lambda _client=None: _snapshot(list(reversed(tickers)), "Wikipedia"),
    )

    snapshot = ndx_components.load_current_ndx_components("unused.csv", "unused.json")

    assert snapshot.source == "Nasdaq API (verified against Wikipedia)"
    assert snapshot.warning is None


def test_load_current_components_keeps_nasdaq_authoritative_on_difference(monkeypatch):
    nasdaq_tickers = [f"T{i:03d}" for i in range(100)]
    wikipedia_tickers = nasdaq_tickers[:-1] + ["WIKI"]
    monkeypatch.setattr(
        ndx_components,
        "fetch_nasdaq_components",
        lambda _client=None: _snapshot(nasdaq_tickers, "Nasdaq API"),
    )
    monkeypatch.setattr(
        ndx_components,
        "fetch_wikipedia_components",
        lambda _client=None: _snapshot(wikipedia_tickers, "Wikipedia"),
    )

    snapshot = ndx_components.load_current_ndx_components("unused.csv", "unused.json")

    assert snapshot.source == "Nasdaq API"
    assert "Nasdaq remains authoritative" in snapshot.warning
    assert "T099" in snapshot.warning
    assert "WIKI" in snapshot.warning


def test_load_static_components_returns_percent_units(tmp_path):
    component_path = tmp_path / "components.csv"
    mapping_path = tmp_path / "mapping.json"
    rows = [
        {
            "Date": "2026-01-30",
            "FilingID": "test",
            "Company": f"Company {i}",
            "Shares": 1,
            "Value": i + 1,
        }
        for i in range(100)
    ]
    pd.DataFrame(rows).to_csv(component_path, index=False)
    mapping_path.write_text(
        json.dumps({f"Company {i}": f"T{i:03d}" for i in range(100)}),
        encoding="utf-8",
    )

    snapshot = ndx_components.load_static_components(component_path, mapping_path)

    assert snapshot.as_of == pd.Timestamp("2026-01-30")
    assert snapshot.components["Weight"].sum() == pytest.approx(100.0)
    assert snapshot.components.iloc[0]["Ticker"] == "T099"
