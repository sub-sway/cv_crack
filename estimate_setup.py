"""User-confirmed illustrative estimate inputs, persisted separately from sourced prices."""
import copy
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

SCENARIO_VERSION = "illustrative-estimate-v1"
COMPARISON_VERSION = "depth-comparison-v1"
DEPTH_SCENARIOS_CM = (1, 3, 5, 10)
# Deliberately hypothetical starting inputs, not market prices or measured dimensions.
DEFAULT_INPUTS = {
    "method": "injection", "plasterer_daily": 270000,
    "skilled_daily": 220000, "general_daily": 170000,
    "material_krw_per_kg": 25000, "depth_mm": 50,
    "width_mode": "assumed", "width_mm": 0.3,
    "density_kg_per_l": 1.1, "usage_factor": 1.2,
    "surface_kg_per_m": 0.1, "misc_percent": 0,
    "note": "초기 예산 검토용 가정값 — 현장 견적·실측값 아님",
}
LIMITS = {
    "plasterer_daily": (1, 10000000), "skilled_daily": (1, 10000000),
    "general_daily": (1, 10000000), "material_krw_per_kg": (1, 100000000),
    "depth_mm": (0.1, 2000), "width_mm": (0.01, 10),
    "density_kg_per_l": (0.01, 10), "usage_factor": (1, 10),
    "surface_kg_per_m": (0.0001, 100), "misc_percent": (0, 5),
}


def validate_inputs(values):
    import math
    if not isinstance(values, dict):
        raise ValueError("입력값 형식이 올바르지 않습니다.")
    result = {}
    for key, (low, high) in LIMITS.items():
        value = values.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key}: {low}~{high} 범위의 숫자를 입력하세요.")
        result[key] = value
    for key, choices in {"method": ("injection", "surface_treatment"),
                         "width_mode": ("assumed", "observed")}.items():
        if values.get(key) not in choices:
            raise ValueError(f"{key}: 선택값을 확인하세요.")
        result[key] = values[key]
    note = values.get("note", "")
    if not isinstance(note, str) or len(note) > 1000:
        raise ValueError("메모는 1,000자 이내로 입력하세요.")
    result["note"] = note.strip()
    return result


class EstimateSetupStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            if not self.path.exists():
                return {"configured": False, "inputs": copy.deepcopy(DEFAULT_INPUTS)}
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("version") != SCENARIO_VERSION:
                raise ValueError("저장된 추정 설정 형식을 확인하세요.")
            return {"configured": True, "inputs": validate_inputs(data.get("inputs")),
                    "saved_at": data.get("saved_at")}

    def save(self, inputs):
        clean = validate_inputs(inputs)
        data = {"version": SCENARIO_VERSION, "inputs": clean,
                "saved_at": datetime.now().isoformat(timespec="seconds")}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix="estimate_", suffix=".tmp", delete=False) as file:
                    temporary = file.name
                    json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return {"configured": True, "inputs": clean, "saved_at": data["saved_at"]}


def estimate_scenario(length_mm, mean_width_mm, setup, length_valid, width_valid):
    from repair_costs import METHODS, STANDARD_SOURCE, EXCLUSIONS, positive
    result = {"method_key": None, "method_label": "추정 설정 대기",
              "length_m": length_mm / 1000 if positive(length_mm) else 0,
              "unit_price_krw_per_m": None, "estimated_cost_krw": None,
              "model_version": SCENARIO_VERSION, "status": "추정 설정을 먼저 저장하세요.",
              "missing": [], "breakdown": None, "basis": None, "exclusions": list(EXCLUSIONS)}
    if not setup.get("configured"):
        return result
    values = validate_inputs(setup["inputs"])
    method = METHODS[values["method"]]
    result.update(method_key=values["method"], method_label=method["label"] + " (가정)")
    if not length_valid or not positive(length_mm):
        result["status"] = "산출 보류: 유효한 균열 길이 측정 필요"
        return result
    width = values["width_mm"] if values["width_mode"] == "assumed" else mean_width_mm
    if values["method"] == "injection" and (not positive(width) or width > 10 or
            (values["width_mode"] == "observed" and not width_valid)):
        result["status"] = "산출 보류: 관측 폭의 측정 한계 확인 또는 가정 폭 입력 필요"
        return result
    wages = {"plasterer": values["plasterer_daily"], "skilled_laborer": values["skilled_daily"],
             "general_laborer": values["general_daily"]}
    labor = sum(wages[key] * count for key, count in method["crew"].items()) / method["output_m_per_day"]
    quantity = values["surface_kg_per_m"]
    if values["method"] == "injection":
        # Uniform rectangular opening per metre: mm * mm / 1000 = litres/metre.
        quantity = width * values["depth_mm"] / 1000 * values["density_kg_per_l"] * values["usage_factor"]
    material = quantity * values["material_krw_per_kg"]
    unit = {"labor": labor, "tools": labor * method["tool_rate"], "material": material,
            "misc_material": material * values["misc_percent"] / 100}
    result.update({"unit_price_krw_per_m": sum(unit.values()),
                   "estimated_cost_krw": sum(unit.values()) * result["length_m"],
                   "breakdown": {key: value * result["length_m"] for key, value in unit.items()},
                   "status": "가정값 기반 추정 소계 · 내부 깊이 미측정 · 실제 견적 아님",
                   "basis": {"standard": copy.deepcopy(STANDARD_SOURCE), "method": copy.deepcopy(method),
                             "scenario_inputs": values, "assumed_depth_mm": values["depth_mm"],
                             "width_used_mm": width if values["method"] == "injection" else None,
                             "material_kg_per_m": quantity, "saved_at": setup.get("saved_at")}})
    return result


def estimate_depth_scenarios(length_mm, mean_width_mm, setup, length_valid, width_valid):
    """Alternative estimates, never additive damages or a measured internal depth."""
    base = estimate_scenario(length_mm, mean_width_mm, setup, length_valid, width_valid)
    base["model_version"] = COMPARISON_VERSION
    base["scenarios"] = []
    if base["estimated_cost_krw"] is None:
        return base
    depths = DEPTH_SCENARIOS_CM if setup["inputs"]["method"] == "injection" else (None,)
    for depth in depths:
        alternative = copy.deepcopy(setup)
        if depth is not None:
            alternative["inputs"]["depth_mm"] = depth * 10
        value = estimate_scenario(length_mm, mean_width_mm, alternative, length_valid, width_valid)
        base["scenarios"].append({
            "assumed_depth_cm": depth, "estimated_cost_krw": value["estimated_cost_krw"],
            "material_kg_per_m": value["basis"]["material_kg_per_m"],
            "breakdown": value["breakdown"],
        })
    base.update(estimated_cost_krw=None, unit_price_krw_per_m=None, breakdown=None)
    base["basis"].pop("assumed_depth_mm", None)
    base["basis"].pop("material_kg_per_m", None)
    base["basis"]["scenario_inputs"].pop("depth_mm", None)
    base["status"] = ("내부 깊이 미측정 · 1·3·5·10cm 가정별 추정 비용 (서로 합산하지 않음)"
                      if depths != (None,) else "표면처리 추정 비용 · 내부 깊이에 따른 비용 차이 미적용")
    return base
