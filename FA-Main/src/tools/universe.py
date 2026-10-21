"""Approved product universe filtering tool (Slide 16)."""

import os
import json
from typing import List, Dict, Any, Optional
from src.schemas.tools import ApprovedInstrument

DATA_UNIVERSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data", "universe", "approved_products.json"
)

def approved_product_universe_tool(asset_filters: Optional[Dict[str, Any]] = None) -> List[ApprovedInstrument]:
    """
    Retrieve institutional approved instruments filtered by client constraints,
    complexity ceiling, and allowed asset classes.
    """
    if not os.path.exists(DATA_UNIVERSE_PATH):
        return []

    with open(DATA_UNIVERSE_PATH, "r", encoding="utf-8") as f:
        products = json.load(f)

    filters = asset_filters or {}
    max_complexity = filters.get("max_complexity", 5)
    allowed_asset_classes = filters.get("allowed_asset_classes", [])
    excluded_sectors = [s.lower() for s in filters.get("excluded_sectors", [])]
    esg_required = filters.get("esg_mandate", False)

    approved_list: List[ApprovedInstrument] = []

    for item in products:
        if not item.get("is_approved", False) or item.get("status") != "APPROVED":
            continue

        complexity = item.get("complexity_rating", 1)
        if complexity > max_complexity:
            continue

        asset_class = item.get("asset_class", "")
        if allowed_asset_classes and asset_class not in allowed_asset_classes:
            continue

        sub_asset = item.get("sub_asset_class", "").lower()
        if any(sec in sub_asset for sec in excluded_sectors):
            continue

        if esg_required:
            esg_rating = item.get("esg_rating", "")
            if esg_rating in ["BBB", "BB", "B", "CCC", "N/A", ""]:
                continue

        approved_list.append(ApprovedInstrument(
            instrument_id=item["instrument_id"],
            ticker=item["ticker"],
            name=item["name"],
            asset_class=item["asset_class"],
            sub_asset_class=item["sub_asset_class"],
            region=item["region"],
            currency=item["currency"],
            risk_rating=item["risk_rating"],
            complexity_rating=complexity,
            liquidity_tier=item["liquidity_tier"],
            is_approved=item["is_approved"],
            status=item["status"],
            esg_rating=item.get("esg_rating"),
            sector=item.get("sector", "Diversified"),
            instrument_type=item.get("instrument_type", "Fund")
        ))

    return approved_list
