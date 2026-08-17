from screener.results import load_screened_results


def test_canonical_screener_results_are_available() -> None:
    results = load_screened_results()

    assert all("CTA" not in result["local"]["allocation"] for result in results)
    assert all(not result["gate_reasons"] for result in results)
