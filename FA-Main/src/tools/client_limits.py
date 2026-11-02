"""Client limits layer: what UBS can offer each client (deck slide 12).

A rule layer, not an AI judgement. It maps the client's regulatory profile against the approved product
universe BEFORE any briefing or idea is produced, and attaches the required disclosures. A blocked product is
never shown to the RM; a gap is listed, never guessed. Shared by UC1 and UC2; the Fit & Eligibility Critic
reuses it in UC3 to check spoken profile changes.
"""

import copy
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.schemas.advisory import BlockedArea, LimitGap, OfferMap, OfferedProduct, ProfileChangeCheck
from src.tools.profile_tools import risk_profile_calculator
from src.tools.risk_engine import DEFAULT_REFERENCE_DATE, load_policy
from src.tools.universe import approved_product_universe_tool

REGULATORY_FIELDS = ["domicile", "booking_centre", "agreement_type", "investor_classification"]
# Spoken profile-change fields that are regulatory facts (re-run through the limits layer) vs risk facts.
REGULATORY_CHANGE_FIELDS = REGULATORY_FIELDS + ["pep_status"]
RISK_CHANGE_FIELDS = ["time_horizon_months", "stated_risk_willingness"]
CAPACITY_CHANGE_FIELDS = ["income", "employment", "liquidity_need"]

def _policy() -> Dict[str, Any]:
    return load_policy()["client_limits"]

def _ref_date(reference_date_iso: str) -> date:
    return datetime.fromisoformat(reference_date_iso.replace("Z", "+00:00")).date()

def _parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None

def _expiry(valid_until: Any, today: date, soon_days: int) -> str:
    """'missing' | 'expired' | 'expiring' | 'valid'."""
    d = _parse_date(valid_until)
    if d is None:
        return "missing"
    if d < today:
        return "expired"
    return "expiring" if (d - today).days <= soon_days else "valid"

def _is_us_person(reg: Dict[str, Any]) -> bool:
    return bool(reg.get("us_person")) or reg.get("domicile") in _policy()["us_person_domiciles"]

def _restriction_hit(rule: Dict[str, str], product: Any) -> bool:
    if "instrument_type" in rule and product.instrument_type == rule["instrument_type"]:
        return True
    if "asset_class" in rule and product.asset_class == rule["asset_class"]:
        return True
    return "sub_asset_contains" in rule and rule["sub_asset_contains"] in product.sub_asset_class.lower()

def client_limits_engine(raw_input: Dict[str, Any], reference_date_iso: str = DEFAULT_REFERENCE_DATE) -> OfferMap:
    """Read the regulatory profile, match it against the product universe, attach disclosures, mark blocked areas and gaps."""
    cfg = _policy()
    reg: Dict[str, Any] = raw_input.get("regulatory_profile") or {}
    today = _ref_date(reference_date_iso)
    soon = cfg["expiring_soon_days"]
    base_ccy = str(raw_input.get("base_currency", "USD")).upper()

    gaps: List[LimitGap] = []
    flags: List[str] = []
    areas: Dict[str, str] = {}  # area -> reason, so one area is listed once
    blocked: Dict[str, str] = {}  # ticker -> area (internal)

    # 1. Mandatory regulatory fields. Without them eligibility cannot be determined, so nothing is offered.
    missing = [f for f in cfg["required_regulatory_fields"] if not reg.get(f)]
    for f in missing:
        gaps.append(LimitGap(code="MISSING_REGULATORY_FIELD", message=f"Regulatory field '{f}' is missing.",
                             resolution=f"Obtain '{f}' from the client record and enter it before products can be offered."))

    # 2. Agreement type: Execution-only clients receive no recommendations.
    agreement = reg.get("agreement_type")
    no_advice = bool(agreement) and agreement not in cfg["agreement_types_with_advice"]
    if no_advice:
        gaps.append(LimitGap(code="NO_ADVICE_AGREEMENT", message=f"Agreement type '{agreement}' does not cover advice.",
                             resolution="Recommendations need an Advisory or Discretionary agreement."))
        flags.append(f"Agreement '{agreement}': no recommendations may be made.")

    # 3. PEP status: escalate; the RM must record the compliance review.
    if reg.get("pep_status"):
        gaps.append(LimitGap(code="PEP_REVIEW", message="Client is a politically exposed person (PEP).",
                             resolution="Enhanced due diligence sign-off must be on file before advice."))
        flags.append("PEP: escalate to Compliance for enhanced due diligence.")

    # 4. CKA validity (specified products) and tax forms (US-domiciled products).
    cka = _expiry(reg.get("cka_valid_until"), today, soon)
    cka_blocks = cka in ("missing", "expired")
    if cka == "missing":
        gaps.append(LimitGap(code="CKA_MISSING", message="No Customer Knowledge Assessment (CKA) on file.",
                             resolution="Complete a CKA before specified investment products can be offered."))
        flags.append("No CKA on file: specified products suspended.")
    elif cka == "expired":
        gaps.append(LimitGap(code="CKA_EXPIRED", message=f"CKA expired on {reg.get('cka_valid_until')}.",
                             resolution="Refresh the CKA before specified investment products can be offered."))
        flags.append("Expired CKA: specified products suspended.")
    elif cka == "expiring":
        gaps.append(LimitGap(code="CKA_EXPIRING", message=f"CKA expires on {reg.get('cka_valid_until')}.",
                             resolution="Schedule a CKA refresh."))

    us_person = _is_us_person(reg)
    form_name = cfg["tax_form_for_us_domiciled_products"]["us_person" if us_person else "other"]
    form = next((f for f in reg.get("tax_forms") or [] if str(f.get("form", "")).upper() == form_name), None)
    form_state = _expiry(form.get("valid_until") if form else None, today, soon)
    form_blocks = form_state in ("missing", "expired")
    if form_state == "missing":
        gaps.append(LimitGap(code="TAX_FORM_MISSING", message=f"Tax form {form_name} is not on file.",
                             resolution=f"Collect a valid {form_name} before US-domiciled products can be offered."))
    elif form_state == "expired":
        gaps.append(LimitGap(code="TAX_FORM_EXPIRED", message=f"Tax form {form_name} expired on {form.get('valid_until')}.",
                             resolution=f"Collect a renewed {form_name}."))
    elif form_state == "expiring":
        gaps.append(LimitGap(code="TAX_FORM_EXPIRING", message=f"Tax form {form_name} expires on {form.get('valid_until')}.",
                             resolution=f"Schedule a {form_name} renewal."))

    # 5. Product-by-product eligibility.
    universe = approved_product_universe_tool()
    cannot_determine = bool(missing) or no_advice
    eligible: List[OfferedProduct] = []
    all_disclosures: List[str] = [cfg["disclosures"]["all"]]
    restrictions = [r for r in reg.get("restrictions") or [] if r in cfg["restriction_rules"]]
    classification = str(reg.get("investor_classification", "")).lower()

    def block(ticker: str, area: str, reason: str) -> None:
        areas.setdefault(area, reason)
        blocked.setdefault(ticker, area)

    for p in universe:
        if cannot_determine:
            block(p.ticker, "All products", "Eligibility cannot be determined until the gaps below are resolved.")
            continue
        if us_person and p.fund_domicile in cfg["us_person_blocked_fund_domiciles"]:
            block(p.ticker, "Non-US-domiciled funds", "Not distributable to US persons.")
        if classification == "retail" and p.complexity_rating > cfg["retail_max_complexity"]:
            block(p.ticker, "Complex products", "Above the retail investor complexity limit.")
        if cka_blocks and p.complexity_rating >= cfg["specified_product_min_complexity"]:
            block(p.ticker, "Specified investment products", "A valid CKA is required.")
        if form_blocks and p.fund_domicile == "US":
            block(p.ticker, "US-domiciled products", f"A valid {form_name} is required.")
        for r in restrictions:
            if _restriction_hit(cfg["restriction_rules"][r], p):
                block(p.ticker, f"Client restriction: {r.replace('_', ' ')}", "Excluded by the client's own restriction.")
        if p.ticker in blocked:
            continue
        disclosures = [cfg["disclosures"]["all"]]
        if p.complexity_rating >= cfg["specified_product_min_complexity"]:
            disclosures.append(cfg["disclosures"]["specified_product"])
        if p.currency != base_ccy:
            disclosures.append(cfg["disclosures"]["fx"])
        if p.asset_class == "Fixed Income":
            disclosures.append(cfg["disclosures"]["fixed_income"])
        eligible.append(OfferedProduct(ticker=p.ticker, name=p.name, required_disclosures=disclosures))
        all_disclosures += [d for d in disclosures if d not in all_disclosures]

    if cannot_determine or not eligible:
        status = "BLOCKED"
        if not cannot_determine:
            gaps.append(LimitGap(code="MISSING_REGULATORY_FIELD", message="No product in the approved universe is eligible for this client.",
                                 resolution="Resolve the blocked areas listed, or escalate."))
    elif flags:
        status = "ESCALATE"
    elif gaps:
        status = "GAPS"
    else:
        status = "CLEAR"

    return OfferMap(
        client_id=str(raw_input.get("client_id", "CL-UNASSIGNED")), status=status,
        regulatory_summary={f: str(reg.get(f, "not provided")) for f in cfg["required_regulatory_fields"]}
        | {"pep_status": "Yes" if reg.get("pep_status") else "No", "cka": cka, f"tax_form_{form_name}": form_state},
        eligible_products=[] if status == "BLOCKED" else eligible,
        required_disclosures=all_disclosures if status != "BLOCKED" else [],
        blocked_areas=[BlockedArea(area=a, reason=r) for a, r in areas.items()],
        gaps=gaps, escalation_flags=flags, blocked_tickers=sorted(blocked) if status != "BLOCKED" else sorted(t.ticker for t in universe),
    )

# ==========================================
# UC3: spoken profile changes, checked against the limits and the risk band
# ==========================================

def _parse_months(text: str) -> Optional[int]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(year|yr|month)", text.lower())
    if not m:
        return None
    n = float(m.group(1))
    return int(round(n * 12)) if m.group(2) in ("year", "yr") else int(round(n))

def apply_profile_change(raw_input: Dict[str, Any], field: str, after: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """A copy of the raw intake with one proposed change applied, or a reason it cannot be applied. Never mutates the original."""
    updated = copy.deepcopy(raw_input)
    reg = updated.setdefault("regulatory_profile", {})
    if field in ("domicile", "booking_centre", "agreement_type", "investor_classification"):
        reg[field] = after
    elif field == "pep_status":
        reg[field] = after.strip().lower() in ("yes", "true", "pep", "1")
    elif field == "time_horizon_months":
        months = _parse_months(after) or (int(after) if after.strip().isdigit() else None)
        if months is None:
            return updated, f"Could not read a horizon from '{after}'."
        updated[field] = months
    elif field == "stated_risk_willingness":
        updated[field] = after
    else:
        return updated, None
    return updated, None

def profile_change_limits_check(raw_input: Dict[str, Any], field: str, after: str,
                                reference_date_iso: str = DEFAULT_REFERENCE_DATE) -> ProfileChangeCheck:
    """The Fit & Eligibility Critic's check of one spoken change. It FLAGs for the RM; it never applies the change."""
    updated, problem = apply_profile_change(raw_input, field, after)
    if problem:
        return ProfileChangeCheck(status="FLAG", implications=[problem + " The RM must enter the value before it can be checked."])

    if field in REGULATORY_CHANGE_FIELDS:
        before_map = client_limits_engine(raw_input, reference_date_iso)
        after_map = client_limits_engine(updated, reference_date_iso)
        new_areas = [a.area for a in after_map.blocked_areas if a.area not in {b.area for b in before_map.blocked_areas}]
        new_gaps = [g.message for g in after_map.gaps if g.code not in {b.code for b in before_map.gaps}]
        new_flags = [f for f in after_map.escalation_flags if f not in before_map.escalation_flags]
        lost = len(before_map.eligible_products) - len(after_map.eligible_products)
        implications = [f"What UBS can offer changes from {len(before_map.eligible_products)} to {len(after_map.eligible_products)} eligible products."] if lost else []
        status = "FAIL" if after_map.status == "BLOCKED" and before_map.status != "BLOCKED" else (
            "FLAG" if new_areas or new_gaps or new_flags or lost else "PASS")
        return ProfileChangeCheck(status=status, implications=implications or ["No change to what UBS can offer this client."],
                                  new_gaps=new_gaps, newly_blocked_areas=new_areas, escalation_flags=new_flags)

    if field in RISK_CHANGE_FIELDS:
        before_calc, after_calc = risk_profile_calculator(raw_input), risk_profile_calculator(updated)
        implications = []
        if before_calc.risk_tier != after_calc.risk_tier:
            implications.append(f"Risk band moves from {before_calc.risk_tier} to {after_calc.risk_tier}.")
        if after_calc.has_capacity_conflict and not before_calc.has_capacity_conflict:
            implications.append("Risk willingness would now exceed capacity (R5).")
        return ProfileChangeCheck(status="FLAG" if implications else "PASS",
                                  implications=implications or ["Risk band is unchanged."])

    # Income, employment, liquidity and anything else: capacity may change but cannot be calculated from speech.
    return ProfileChangeCheck(status="FLAG", implications=[
        "This may change the client's risk capacity. Confirm the new figures and re-run the profile before relying on the current risk band."])
