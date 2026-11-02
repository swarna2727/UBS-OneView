"""Client limits layer (deck slide 12): a rule layer that maps what UBS can offer before any briefing or idea."""

import copy
import pytest
from src.tools.client_limits import client_limits_engine, profile_change_limits_check


def sarah(sample_profiles, **reg):
    raw = copy.deepcopy(sample_profiles["demo_sarah"])
    raw["regulatory_profile"].update(reg)
    return raw


def test_clear_client_gets_eligible_products_with_disclosures(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles))
    assert om.status == "CLEAR" and om.gaps == [] and om.escalation_flags == []
    tickers = om.eligible_tickers
    assert "IEF" in tickers and "IEAC" in tickers
    assert "Product risk disclosure" in om.required_disclosures
    iec = next(p for p in om.eligible_products if p.ticker == "IEAC")
    assert "Currency risk disclosure" in iec.required_disclosures  # EUR product for an SGD client


def test_product_blocked_by_domicile_is_never_named_to_the_rm(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, domicile="US", tax_forms=[{"form": "W-9", "valid_until": "2030-01-01"}]))
    assert {"IEAC", "AGGG"} <= set(om.blocked_tickers) and "IEAC" not in om.eligible_tickers
    assert any(a.area == "Non-US-domiciled funds" for a in om.blocked_areas)
    shown = str(om.rm_view())
    assert "IEAC" not in shown and "blocked_tickers" not in shown  # areas and reasons, never the product


def test_expired_cka_suspends_specified_products_escalates_and_lists_the_gap(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, cka_valid_until="2026-01-01"))
    assert om.status == "ESCALATE"
    assert {"HYG", "TSM", "VNQ"} <= set(om.blocked_tickers)
    assert [g.code for g in om.gaps] == ["CKA_EXPIRED"] and om.escalation_flags


def test_pep_is_escalated_with_a_gap_but_products_stay_eligible(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, pep_status=True))
    assert om.status == "ESCALATE" and any(g.code == "PEP_REVIEW" for g in om.gaps)
    assert "IEF" in om.eligible_tickers


def test_missing_tax_form_blocks_only_us_domiciled_products(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, tax_forms=[]))
    assert any(g.code == "TAX_FORM_MISSING" for g in om.gaps)
    assert "IEF" in om.blocked_tickers and "IEAC" in om.eligible_tickers


def test_missing_regulatory_field_is_a_gap_never_guessed(sample_profiles):
    raw = sarah(sample_profiles)
    del raw["regulatory_profile"]["domicile"]
    om = client_limits_engine(raw)
    assert om.status == "BLOCKED" and om.eligible_products == []
    assert any(g.code == "MISSING_REGULATORY_FIELD" and "domicile" in g.message for g in om.gaps)


def test_no_regulatory_profile_at_all_blocks_everything(sample_profiles):
    raw = copy.deepcopy(sample_profiles["demo_sarah"])
    del raw["regulatory_profile"]
    om = client_limits_engine(raw)
    assert om.status == "BLOCKED" and sum(g.code == "MISSING_REGULATORY_FIELD" for g in om.gaps) == 4


def test_execution_only_agreement_receives_no_recommendations(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, agreement_type="Execution-only"))
    assert om.status == "BLOCKED" and any(g.code == "NO_ADVICE_AGREEMENT" for g in om.gaps)


def test_client_restriction_blocks_that_area(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, restrictions=["no_high_yield"]))
    assert "HYG" in om.blocked_tickers and any("high yield" in a.area for a in om.blocked_areas)


def test_expiring_cka_is_listed_but_does_not_block(sample_profiles):
    om = client_limits_engine(sarah(sample_profiles, cka_valid_until="2026-11-30"))
    assert om.status == "GAPS" and [g.code for g in om.gaps] == ["CKA_EXPIRING"] and "HYG" in om.eligible_tickers


def test_spoken_domicile_change_is_checked_against_limits_without_being_applied(sample_profiles):
    raw = sarah(sample_profiles)
    check = profile_change_limits_check(raw, "domicile", "US")
    assert check.status == "FLAG" and "Non-US-domiciled funds" in check.newly_blocked_areas
    assert raw["regulatory_profile"]["domicile"] == "Singapore"  # the check never mutates the client record


def test_spoken_income_change_is_flagged_for_the_rm(sample_profiles):
    check = profile_change_limits_check(sarah(sample_profiles), "income", "May change from next month")
    assert check.status == "FLAG" and "risk capacity" in check.implications[0]


def test_unreadable_horizon_is_flagged_not_guessed(sample_profiles):
    check = profile_change_limits_check(sarah(sample_profiles), "time_horizon_months", "soonish")
    assert check.status == "FLAG" and "Could not read" in check.implications[0]
