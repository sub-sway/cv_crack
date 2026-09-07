"""Width changes between tracked observations; not proof of structural deterioration."""
import math

WIDTH_VERSION = "skeleton-distance-v1"

# 고해상도 재측정본은 축척이 3배 다르므로 저해상도 관측과 섞어 비교하면 안 됩니다.
# comparable()이 dimension_scale_source 동일성을 요구하므로 이름만 구분해 두면
# 서로 다른 축척끼리는 자동으로 비교 보류가 됩니다.
SCALE_SOURCES = (
    "aruco", "depth_approximation",
    "aruco_highres", "depth_approximation_highres",
)


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def width_sample(record):
    return {key: record.get(key) for key in (
        "timestamp", "width_profile_version", "width_p95_mm", "raw_max_width_mm",
        "mean_width_mm", "mm_per_pixel", "dimension_scale_source", "aruco_marker_id",
        "scale_stale", "grade", "widest_point_px", "measurement_limit_mm",
    )}


def limit_mm(sample):
    """판정 한계. 고해상도 재측정본은 2px가 아니라 4px 기준입니다."""
    recorded = sample.get("measurement_limit_mm")
    return recorded if valid_number(recorded) else 2 * sample["mm_per_pixel"]


def usable(sample):
    return (valid_number(sample.get("width_p95_mm"))
            and valid_number(sample.get("mm_per_pixel"))
            and sample["width_p95_mm"] > limit_mm(sample)
            and sample.get("dimension_scale_source") in SCALE_SOURCES
            and not sample.get("scale_stale", False))


def comparable(first, current):
    if not usable(first) or not usable(current):
        return False
    if first.get("width_profile_version") != current.get("width_profile_version"):
        return False
    if first["dimension_scale_source"] != current["dimension_scale_source"]:
        return False
    if first["dimension_scale_source"] == "aruco" and first.get("aruco_marker_id") != current.get("aruco_marker_id"):
        return False
    # Project quality gate, not a metrological uncertainty specification.
    return abs(current["mm_per_pixel"] / first["mm_per_pixel"] - 1) <= 0.10


def record_width_change(record, previous=None):
    previous = previous or {}
    sample = width_sample(record)
    baseline = previous.get("width_baseline")
    new_baseline = baseline is None and usable(sample)
    if new_baseline:
        baseline = dict(sample)
    prior = previous.get("width_last_sample")
    comparable_base = baseline is not None and comparable(baseline, sample)
    comparable_previous = prior is not None and comparable(prior, sample)
    delta = round(sample["width_p95_mm"] - baseline["width_p95_mm"], 4) if comparable_base else None
    limit = limit_mm(sample) + limit_mm(baseline) if comparable_base else None
    if not usable(sample):
        status = "비교 보류: 측정 한계 또는 보정값 확인 필요"
    elif not comparable_base:
        status = "비교 보류: 초기 관측과 보정 방식·축척이 다름"
    elif new_baseline:
        status = "초기 기준 관측"
    elif abs(delta) <= limit:
        status = "해상도 기반 비교 한계 이내 (성장 판정 불가)"
    else:
        status = "폭 추정값 증가 · 동일 위치 재확인 필요" if delta > 0 else "폭 추정값 감소 · 촬영 조건 재확인 필요"
    sample.update({"width_delta_mm": delta, "width_change_status": status})
    return {
        "width_baseline": baseline, "width_last_sample": sample,
        "width_delta_mm": delta,
        "width_previous_delta_mm": round(sample["width_p95_mm"] - prior["width_p95_mm"], 4) if comparable_previous else None,
        "width_change_limit_mm": round(limit, 4) if limit is not None else None,
        "width_change_status": status,
        "width_sample_count": previous.get("width_sample_count", 0) + 1,
        "width_history": (previous.get("width_history", []) + [sample])[-20:],
    }
