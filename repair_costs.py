"""Evidence-based direct-work subtotals; see REPAIR_COST_BASIS.md."""
import copy
import math
from datetime import date
from estimate_setup import SCENARIO_VERSION, COMPARISON_VERSION, DEPTH_SCENARIOS_CM

MODEL_VERSION = "kict-2026-direct-v1"
STANDARD_SOURCE = {
    "title": "2026 건설공사 표준품셈 · 유지관리부문",
    "url": "https://www.kosca21.or.kr/km_board/upload/km_b_notice/1767187711_03.pdf",
    "pages": "인쇄 867–868쪽 (PDF 923–924쪽)",
}
METHODS = {
    "surface_treatment": {
        "label": "표면처리공법", "section": "1-3-1", "output_m_per_day": 110,
        "crew": {"plasterer": 1}, "tool_rate": 0.02,
    },
    "injection": {
        "label": "에폭시 주입공법", "section": "1-3-2", "output_m_per_day": 28,
        "crew": {"skilled_laborer": 2, "general_laborer": 1}, "tool_rate": 0.02,
    },
}
OCCUPATIONS = {"plasterer": "미장공", "skilled_laborer": "특별인부", "general_laborer": "보통인부"}
EXCLUSIONS = ["고소작업·접근 장비", "현장 조건 및 소규모 작업 보정", "간접비·일반관리비·이윤", "부가가치세"]


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def evidence(value):
    if not isinstance(value, dict):
        return False
    if not all(isinstance(value.get(key), str) and value[key].strip()
               for key in ("title", "reference", "date")):
        return False
    try:
        date.fromisoformat(value["date"])
    except ValueError:
        return False
    return True


def as_dict(value):
    return value if isinstance(value, dict) else {}


def estimate_repair_cost(length_mm, width_mm, config, context=None, measurement_valid=False):
    context = as_dict(context)
    key = context.get("repair_method")
    method = METHODS.get(key) if isinstance(key, str) else None
    result = {
        "method_key": key, "method_label": method["label"] if method else "공법 검토 필요",
        "length_m": length_mm / 1000 if positive(length_mm) else 0,
        "unit_price_krw_per_m": None, "estimated_cost_krw": None,
        "model_version": MODEL_VERSION, "status": "산출 보류",
        "missing": [], "breakdown": None, "basis": None,
        "exclusions": list(EXCLUSIONS),
    }
    missing = result["missing"]
    if not method:
        missing.append("담당자의 보수 공법 선택")
    if not isinstance(context.get("repair_review_note"), str) or not context["repair_review_note"].strip():
        missing.append("공법 적용 검토 근거")
    if context.get("member_material") not in ("rc", "psc", "rc_crossbeam"):
        missing.append("콘크리트 부재 확인")
    if not measurement_valid or not positive(length_mm) or not positive(width_mm):
        missing.append("측정 한계를 초과하는 유효 치수")
    elif width_mm > 10:
        missing.append("균열폭 10mm 초과: 별도 적산")
    config = as_dict(config)
    if config.get("schema_version") != 2 or config.get("currency") != "KRW":
        missing.append("근거를 포함한 새 비용 설정")
    if missing:
        result["status"] += ": " + ", ".join(missing)
        return result

    profile = as_dict(as_dict(config.get("methods")).get(key))
    scope = as_dict(profile.get("scope"))
    if not all(isinstance(scope.get(field), str) and scope[field].strip()
               and scope[field] == context.get(field)
               for field in ("bridge_id", "member_label", "member_position")):
        missing.append("단가·설계수량의 교량 ID/부재/위치 적용 범위 일치")
    wages = as_dict(config.get("labor_rates"))
    daily_labor = 0
    for occupation, count in method["crew"].items():
        item = as_dict(wages.get(occupation))
        if not positive(item.get("krw_per_day")) or not evidence(item.get("source")):
            missing.append(OCCUPATIONS[occupation] + " 일 노임 및 출처·기준일")
        else:
            daily_labor += count * item["krw_per_day"]
    material = as_dict(profile.get("material"))
    if not isinstance(material.get("name"), str) or not material["name"].strip():
        missing.append("주재료 제품명")
    if not positive(material.get("kg_per_m")) or not evidence(material.get("quantity_source")):
        missing.append("설계 주재료 수량(kg/m) 및 근거")
    if not positive(material.get("krw_per_kg")) or not evidence(material.get("price_source")):
        missing.append("주재료 가격(원/kg) 및 출처·기준일")
    rate = profile.get("misc_material_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 0.05:
        missing.append("잡재료율(0~5%) 명시")
    if not evidence(profile.get("misc_material_source")):
        missing.append("잡재료율 선택 근거")
    if missing:
        result["status"] += ": " + ", ".join(missing)
        return result

    labor = daily_labor / method["output_m_per_day"]
    material_cost = material["kg_per_m"] * material["krw_per_kg"]
    unit = {"labor": labor, "tools": labor * method["tool_rate"],
            "material": material_cost, "misc_material": material_cost * rate}
    result.update({
        "unit_price_krw_per_m": sum(unit.values()),
        "estimated_cost_krw": sum(unit.values()) * result["length_m"],
        "status": "등록 근거에 따른 직접작업비 소계 (총공사비 아님)",
        "breakdown": {name: value * result["length_m"] for name, value in unit.items()},
        "basis": copy.deepcopy({"standard": STANDARD_SOURCE, "method": method,
                                 "labor_rates": {name: wages[name] for name in method["crew"]},
                                 "profile": profile, "review_note": context["repair_review_note"]}),
    })
    return result


def present_record(record):
    """Keep historical unsourced estimates out of new report totals without deleting data."""
    record = dict(record)
    if record.get("cost_model_version") not in (MODEL_VERSION, SCENARIO_VERSION, COMPARISON_VERSION):
        record.update({"estimated_repair_cost_krw": None, "unit_price_krw_per_m": None,
                       "recommended_repair_method": "공법 재검토 필요",
                       "cost_status": "산출 보류: 이전 계산은 단가 근거 미확인",
                       "cost_basis": None, "cost_breakdown": None})
    return record


def summarize_costs(records):
    comparisons = []
    if any(r.get("cost_model_version") == COMPARISON_VERSION for r in records):
        available = [s for r in records if r.get("cost_model_version") == COMPARISON_VERSION
                     for s in r.get("cost_scenarios", [])]
        depths = (None,) if available and all(s["assumed_depth_cm"] is None for s in available) else DEPTH_SCENARIOS_CM
        for depth in depths:
            amounts = []
            for record in records:
                if record.get("cost_model_version") != COMPARISON_VERSION:
                    continue
                item = next((s for s in record.get("cost_scenarios", [])
                             if s["assumed_depth_cm"] in (depth, None)), None)
                if item and positive(item.get("estimated_cost_krw")):
                    amounts.append(item["estimated_cost_krw"])
            comparisons.append({"assumed_depth_cm": depth,
                                "estimated_cost_krw": round(sum(amounts)) if amounts and len(amounts) == len(records) else None,
                                "priced_subtotal_krw": round(sum(amounts)) if amounts else None,
                                "unpriced_count": len(records) - len(amounts)})
    priced = [r for r in records if r.get("cost_model_version") in (MODEL_VERSION, SCENARIO_VERSION)
              and positive(r.get("estimated_repair_cost_krw"))]
    partial = round(sum(r["estimated_repair_cost_krw"] for r in priced)) if priced else None
    return {"estimated_cost_krw": partial if priced and len(priced) == len(records) else None,
            "priced_subtotal_krw": partial, "priced_count": len(priced),
            "unpriced_count": len(records) - len(priced),
            "assumed_count": sum(r.get("cost_model_version") == SCENARIO_VERSION for r in priced),
            "depth_scenarios": comparisons}
