"use strict";

const POLL_INTERVAL_MS = 500;
const REPORT_INTERVAL_MS = 5000;
const STREAM_URL = "/video_feed";

let currentPage = "dashboard";
let pollTimer = null;
let reportTimer = null;
let pollDelay = POLL_INTERVAL_MS;
let dangerOnly = false;

const $ = (id) => document.getElementById(id);
const ESTIMATE_NUMBERS = ["plasterer_daily", "skilled_daily", "general_daily", "material_krw_per_kg", "depth_mm", "width_mm", "density_kg_per_l", "usage_factor", "surface_kg_per_m", "misc_percent"];
let estimateDefaults = null;
let estimateNorms = null;
let estimateDepths = [1, 3, 5, 10];
let estimateLoading = false;

function estimateInputs() {
    const values = {};
    ESTIMATE_NUMBERS.forEach(key => { values[key] = $("e-" + key).value === "" ? null : Number($("e-" + key).value); });
    ["method", "width_mode", "note"].forEach(key => { values[key] = $("e-" + key).value; });
    return values;
}

function fillEstimate(values) {
    [...ESTIMATE_NUMBERS, "method", "width_mode", "note"].forEach(key => { $("e-" + key).value = values[key] ?? ""; });
    previewEstimate();
}

function previewEstimate() {
    const values = estimateInputs();
    document.querySelectorAll("[data-estimate-method]").forEach(label => {
        label.hidden = label.dataset.estimateMethod !== values.method;
        label.querySelectorAll("input,select").forEach(field => { field.disabled = label.hidden; });
    });
    const assumed = values.width_mode === "assumed";
    $("estimate-width-label").hidden = values.method !== "injection" || !assumed;
    $("e-width_mm").disabled = $("estimate-width-label").hidden;
    // 숨긴 항목의 미완성 입력이 현재 공법 저장을 막지 않게 합니다.
    $("estimate-form").querySelectorAll("input:disabled").forEach(field => {
        const number = Number(field.value);
        if (field.value === "" || !Number.isFinite(number) || number < Number(field.min) || number > Number(field.max)) {
            field.value = estimateDefaults[field.id.slice(2)];
        }
    });
    const norm = estimateNorms?.[values.method];
    if (!norm) return;
    if (!$("estimate-form").checkValidity()) {
        setText("estimate-preview", "입력 범위에 맞는 값을 채우면 1m 기준 예시 소계를 표시합니다.");
        return;
    }
    const wages = {plasterer: values.plasterer_daily, skilled_laborer: values.skilled_daily, general_laborer: values.general_daily};
    const labor = Object.entries(norm.crew).reduce((sum, [key, count]) => sum + count * wages[key], 0) / norm.output_m_per_day;
    if (values.method === "injection" && !assumed) {
        setText("estimate-preview", "내부 깊이 1·3·5·10cm 가정 · 관측 폭이 들어오면 깊이별로 계산합니다. 측정 한계 이하의 관측 폭은 산출을 보류합니다.");
        return;
    }
    const depths = values.method === "injection" ? estimateDepths : [null];
    const rows = depths.map(depth => {
        const quantity = depth === null ? values.surface_kg_per_m : values.width_mm * depth * 10 / 1000 * values.density_kg_per_l * values.usage_factor;
        const material = quantity * values.material_krw_per_kg;
        return {assumed_depth_cm: depth, estimated_cost_krw: labor * (1 + norm.tool_rate) + material * (1 + values.misc_percent / 100)};
    });
    $("estimate-preview").innerHTML = "<p>균열 길이 1m 기준 예시 · 각 행은 별개의 가정입니다.</p>" + depthCostTable(rows);
}

function oneMeterEstimateRows() {
    if (!estimateNorms) return null;
    const values = estimateInputs();
    const norm = estimateNorms[values.method];
    if (!norm || !$('estimate-form').checkValidity()) return null;
    if (values.method === 'injection' && values.width_mode !== 'assumed') return null;
    const wages = {plasterer: values.plasterer_daily, skilled_laborer: values.skilled_daily, general_laborer: values.general_daily};
    const labor = Object.entries(norm.crew).reduce((sum, [key, count]) => sum + count * wages[key], 0) / norm.output_m_per_day;
    const depths = values.method === 'injection' ? estimateDepths : [null];
    return depths.map(depth => {
        const quantity = depth === null ? values.surface_kg_per_m : values.width_mm * depth * 10 / 1000 * values.density_kg_per_l * values.usage_factor;
        const material = quantity * values.material_krw_per_kg;
        return {assumed_depth_cm: depth, estimated_cost_krw: labor * (1 + norm.tool_rate) + material * (1 + values.misc_percent / 100)};
    });
}

function renderOneMeterEstimate(reason = '현재 탐지된 균열이 없습니다.') {
    const rows = oneMeterEstimateRows();
    if (!rows) {
        setText('repair-cost', estimateNorms ? '관측 폭 또는 유효한 추정 기준이 필요합니다.' : '추정 기준을 불러오는 중입니다.');
        setText('repair-status', reason);
        return;
    }
    const values = estimateInputs();
    const norm = estimateNorms[values.method];
    setText('repair-method', `${norm.label} (가정)`);
    $('repair-cost').innerHTML = '<p class="stat-note">균열 미탐지 시 표시하는 길이 1m 기준 예시입니다.</p>' + depthCostTable(rows);
    setText('repair-status', `${reason} 내부 깊이는 미측정이며 1·3·5·10cm 각 행은 서로 다른 가정입니다.`);
}

async function loadEstimateSetup(forceOpen = false) {
    if (estimateLoading) return;
    estimateLoading = true;
    if (forceOpen && !$("estimate-dialog").open) $("estimate-dialog").showModal();
    $("estimate-loading").hidden = false;
    $("estimate-form").hidden = true;
    $("estimate-retry").hidden = true;
    setText("estimate-error", "");
    try {
        const response = await fetch("/api/estimate-setup", {cache: "no-store"});
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.message || "추정 설정을 불러오지 못했습니다.");
        estimateDefaults = payload.defaults;
        estimateNorms = payload.norms;
        estimateDepths = payload.depth_scenarios_cm || [1, 3, 5, 10];
        fillEstimate(payload.inputs);
        renderOneMeterEstimate();
        $("estimate-form").hidden = false;
        setText("estimate-save", payload.configured ? "변경한 기준 저장" : "이 값으로 저장하고 시작");
        if ((!payload.configured || forceOpen) && !$("estimate-dialog").open) $("estimate-dialog").showModal();
    } catch (error) {
        setText("estimate-error", error.message + " 비용 추정은 설정 저장 후 사용할 수 있습니다.");
        $("estimate-retry").hidden = false;
        if (!$("estimate-dialog").open) $("estimate-dialog").showModal();
    } finally {
        $("estimate-loading").hidden = true;
        estimateLoading = false;
    }
}

$("estimate-form").addEventListener("input", previewEstimate);
$("estimate-form").addEventListener("change", previewEstimate);
$("estimate-defaults").addEventListener("click", () => fillEstimate(estimateDefaults));
$("estimate-close").addEventListener("click", () => $("estimate-dialog").close());
$("estimate-retry").addEventListener("click", () => loadEstimateSetup(true));
$("estimate-form").addEventListener("submit", async event => {
    event.preventDefault();
    const button = $("estimate-save");
    button.disabled = true;
    setText("estimate-error", "");
    try {
        const response = await fetch("/api/estimate-setup", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({inputs: estimateInputs()})});
        const payload = await response.json();
        if (!response.ok || !payload.success) throw new Error(payload.message || "저장하지 못했습니다.");
        $("estimate-dialog").close();
        showToast("추정 기준을 저장했습니다. 이후 관측에 적용됩니다.");
    } catch (error) {
        setText("estimate-error", error.message);
    } finally {
        button.disabled = false;
    }
});

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
    }[char]));
}

function setText(id, value) {
    const element = $(id);
    if (element) {
        element.textContent = value;
    }
}

function formatNumber(value, digits = 1, unit = "") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
    }
    return Number(value).toFixed(digits) + unit;
}

function formatCost(value) {
    if (value === null || value === undefined) {
        return "단가 미설정";
    }
    return Number(value).toLocaleString() + " 원";
}

function scaleLabel(crack) {
    const labels = {
        aruco: "ArUco 50 mm 기준",
        depth_approximation: "깊이 기반 근사",
    };
    let text = labels[crack.dimension_scale_source] ?? "산출 불가";
    if (crack.below_resolution) {
        text += " · 폭이 측정 한계 이하";
    }
    return text;
}

function riskClass(code) {
    return "risk-" + (["danger", "caution", "normal"].includes(code) ? code : "pending");
}

function badgeClass(code) {
    if (code === "danger") return "status-badge bg-danger";
    if (code === "caution") return "status-badge bg-warning";
    if (code === "normal") return "status-badge bg-success";
    return "status-badge bg-gray";
}

let toastTimer = null;

function showToast(message, isError = false) {
    const toast = $("toast");
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.hidden = true; }, 4000);
}

// ----------------------------------------------------
// 화면 전환 — 보이는 페이지의 영상만 연결합니다.
// 숨은 <img>까지 스트림을 열어두면 서버가 같은 프레임을
// 여러 번 인코딩하느라 분석 FPS가 떨어집니다.
// ----------------------------------------------------

function syncStreams() {
    const active = document.hidden ? null : $(currentPage);
    document.querySelectorAll("img[data-stream]").forEach((image) => {
        const shouldPlay = active && active.contains(image)
            && image.closest(".tab-content, .page").classList.contains("active");

        if (shouldPlay && !image.src) {
            image.src = STREAM_URL + "?t=" + Date.now();
        } else if (!shouldPlay && image.src) {
            image.removeAttribute("src");
        }
    });
}

function switchPage(pageId) {
    currentPage = pageId;

    document.querySelectorAll(".page").forEach((page) => {
        page.classList.toggle("active", page.id === pageId);
    });
    document.querySelectorAll(".nav-item").forEach((item) => {
        if (item.dataset.page === pageId) {
            item.setAttribute("aria-current", "page");
        } else {
            item.removeAttribute("aria-current");
        }
    });

    syncStreams();
    scheduleReport(pageId === "report");
}

function switchTab(tabId) {
    document.querySelectorAll(".tab-content").forEach((content) => {
        content.classList.toggle("active", content.id === tabId);
    });
    document.querySelectorAll(".tab").forEach((tab) => {
        tab.setAttribute("aria-selected", String(tab.dataset.tab === tabId));
    });
    syncStreams();
}

// ----------------------------------------------------
// 실시간 갱신 (요청이 끝난 뒤 다음 요청 예약)
// ----------------------------------------------------

async function pollLatest() {
    try {
        const response = await fetch("/api/latest", { cache: "no-store" });
        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }

        renderLatest(await response.json());
        setText("server-status", "정상");
        pollDelay = POLL_INTERVAL_MS;
    } catch (error) {
        console.error("API 연결 오류:", error);
        setText("server-status", "연결 끊김");

        const badge = $("camera-status");
        badge.textContent = "서버 연결 오류";
        badge.className = "status-badge bg-danger";

        // 서버가 죽었을 때 요청이 쌓이지 않도록 간격을 늘립니다.
        pollDelay = Math.min(pollDelay * 2, 5000);
    } finally {
        pollTimer = setTimeout(pollLatest, pollDelay);
    }
}

function renderLatest(data) {
    const badge = $("camera-status");
    if (data.camera_status === "online") {
        badge.textContent = "정상";
        badge.className = "status-badge bg-success";
    } else if (data.camera_status === "error") {
        badge.textContent = "오류";
        badge.className = "status-badge bg-danger";
    } else {
        badge.textContent = "연결 중";
        badge.className = "status-badge bg-warning";
    }

    setText("usb-speed", data.usb_speed ?? "-");
    setText("fps-value", data.fps != null ? data.fps + " FPS" : "-");
    setText("inference-value", data.inference_ms != null ? data.inference_ms + " ms" : "-");
    setText("last-updated", data.timestamp ?? "-");

    if (data.aruco_detected) {
        setText(
            "aruco-status",
            `ArUco ${data.aruco_marker_id} · ${data.aruco_scale_mm_per_pixel} mm/px`
                + (data.aruco_stale ? " (직전 값 유지)" : "")
        );
    } else {
        setText("aruco-status", "깊이 기반 근사");
    }

    setText(
        "measurement-limit",
        data.measurement_limit_mm != null
            ? `약 ${formatNumber(data.measurement_limit_mm, 2)} mm 이상`
            : "-"
    );

    const detections = data.detections || [];
    setText("dashboard-crack-count", detections.length + " 건");
    setText("detection-crack-count", detections.length);

    const dangerous = detections.filter((crack) => crack.risk_code === "danger");
    const dangerText = $("dashboard-danger-count");
    if (dangerous.length > 0) {
        dangerText.textContent = `위험 균열 ${dangerous.length}건`;
        dangerText.className = "stat-note risk-danger";
    } else {
        dangerText.textContent = "위험 균열 없음";
        dangerText.className = "stat-note risk-normal";
    }

    renderWarnings(data, dangerous);

    if (detections.length === 0) {
        clearCrackDetail();
    } else {
        const representative = [...detections].sort(
            (a, b) => (b.max_width_mm ?? 0) - (a.max_width_mm ?? 0)
        )[0];
        renderCrackDetail(representative);
    }

    renderCurrentTable(detections);
}

function renderWarnings(data, dangerous) {
    const list = $("warning-list");

    if (data.camera_status === "error") {
        list.innerHTML = `<li class="risk-danger">
            카메라 오류: ${escapeHtml(data.error ?? "원인 미상")} — 케이블과 전원을 확인하세요.
        </li>`;
        return;
    }

    if (dangerous.length === 0) {
        list.innerHTML = '<li class="risk-normal">위험 균열 없음</li>';
        return;
    }

    list.innerHTML = dangerous.map((crack) => `
        <li class="risk-danger">
            ${escapeHtml(crack.id)} · 폭 ${formatNumber(crack.max_width_mm, 2, " mm")}
            · ${escapeHtml(crack.risk_level)}
        </li>
    `).join("");
}

function clearCrackDetail() {
    [
        "crack-id", "confidence", "crack-length", "crack-width", "crack-mean-width", "crack-raw-max-width",
        "crack-area", "crack-depth", "depth-valid-ratio", "depth-recessed-ratio",
        "depth-status", "scale-source", "repair-method",
        "dashboard-max-width", "dashboard-depth", "dashboard-depth-status",
    ].forEach((id) => setText(id, "-"));

    const badge = $("risk-badge");
    badge.textContent = "-";
    badge.className = "status-badge bg-gray";
    renderOneMeterEstimate();
}

function renderCrackDetail(crack) {
    setText("crack-id", crack.id);
    setText("confidence", formatNumber(crack.confidence * 100, 1, "%"));
    setText("crack-length", formatNumber(crack.length_mm, 1, " mm"));
    setText("crack-width", formatNumber(crack.max_width_mm, 2, " mm"));
    setText("crack-mean-width", formatNumber(crack.mean_width_mm, 2, " mm"));
    setText("crack-raw-max-width", formatNumber(crack.raw_max_width_mm, 2, " mm"));
    setText("crack-area", formatNumber(crack.area_mm2, 1, " mm²"));
    setText("dashboard-max-width", formatNumber(crack.max_width_mm, 2, " mm"));

    const depthText = crack.depth_diff_mm != null
        ? formatNumber(crack.depth_diff_mm, 1, " mm")
        : "측정 불가";
    setText("crack-depth", depthText);
    setText("dashboard-depth", depthText);
    setText("dashboard-depth-status", crack.depth_status ?? "-");
    setText("depth-status", crack.depth_status ?? "-");
    setText("depth-valid-ratio", formatNumber(crack.depth_valid_ratio * 100, 0, "%"));
    setText("depth-recessed-ratio", formatNumber(crack.depth_recessed_ratio * 100, 0, "%"));

    setText("scale-source", scaleLabel(crack));

    setText("repair-method", crack.recommended_repair_method ?? "-");
    if (crack.cost_scenarios?.length) {
        $("repair-cost").innerHTML = depthCostTable(crack.cost_scenarios);
        setText("repair-status", crack.cost_status ?? "산출 근거 대기");
    } else if (crack.estimated_repair_cost_krw != null) {
        $("repair-cost").innerHTML = escapeHtml(formatCost(crack.estimated_repair_cost_krw));
        setText("repair-status", crack.cost_status ?? "산출 근거 대기");
    } else {
        renderOneMeterEstimate(crack.cost_status ?? "현재 균열의 유효 치수 산출이 보류되었습니다.");
    }

    const badge = $("risk-badge");
    badge.textContent = crack.risk_level ?? "-";
    badge.className = badgeClass(crack.risk_code);
}

function renderCurrentTable(detections) {
    const tbody = $("current-detection-body");

    if (detections.length === 0) {
        tbody.innerHTML = '<tr class="empty-row"><td colspan="8">탐지된 균열이 없습니다.</td></tr>';
        return;
    }

    tbody.innerHTML = detections.map((crack) => `
        <tr>
            <td>${escapeHtml(crack.id)}</td>
            <td>${formatNumber(crack.confidence * 100, 1, "%")}</td>
            <td>${formatNumber(crack.length_mm, 1, " mm")}</td>
            <td>${formatNumber(crack.max_width_mm, 2, " mm")}</td>
            <td>${formatNumber(crack.area_mm2, 0, " mm²")}</td>
            <td>${crack.depth_diff_mm != null ? formatNumber(crack.depth_diff_mm, 1, " mm") : "-"}</td>
            <td>${escapeHtml(crack.depth_status)}</td>
            <td class="${riskClass(crack.risk_code)}">${escapeHtml(crack.risk_level)}</td>
        </tr>
    `).join("");
}

// ----------------------------------------------------
// 보고서 (누적 이력 + 저장된 캡처)
// ----------------------------------------------------

function scheduleReport(enabled) {
    clearTimeout(reportTimer);
    if (!enabled) {
        return;
    }
    loadReport();
}

async function loadReport() {
    clearTimeout(reportTimer);
    await Promise.all([loadHistory(), loadCaptures()]);
    if (currentPage === "report" && !document.hidden) {
        reportTimer = setTimeout(loadReport, REPORT_INTERVAL_MS);
    }
}

async function loadHistory() {
    try {
        const response = await fetch("/api/report", { cache: "no-store" });
        const payload = await response.json();
        const records = payload.records || [];

        renderPrintHeader(payload.inspection, payload.generated_at);
        renderSummary(payload.summary || {}, records);
        renderDamageTable(records);
        renderDamageDetails(records);
    } catch (error) {
        console.error("보고서 조회 오류:", error);
    }
}

function renderPrintHeader(info, generatedAt) {
    if (!info) {
        return;
    }

    setText(
        "print-title",
        (info.bridge_name || "교량") + " 점검 보고서"
    );
    setText(
        "print-subtitle",
        [info.bridge_id, info.site, generatedAt].filter(Boolean).join(" · ")
    );

    const rows = [
        ["교량명", info.bridge_name],
        ["교량 ID", info.bridge_id],
        ["소재지", info.site],
        ["점검일시", generatedAt],
        ["점검자", info.inspector],
        ["기상 상태", info.weather],
        ["촬영 거리", info.capture_distance_mm ? info.capture_distance_mm + " mm" : null],
        ["점검 범위", info.inspection_scope],
        ["점검 부재", [info.member_group, info.member_label || info.member].filter(Boolean).join(" / ")],
        ["부재 재질", MATERIAL_LABELS[info.member_material] ?? info.member_material],
    ];

    $("print-info").innerHTML = rows.map(([label, value]) => `
        <div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value || "-")}</dd></div>
    `).join("");
}

function renderSummary(summary, records) {
    const significant = records.filter(
        (crack) => crack.grade && "cde".includes(crack.grade)
    ).length;

    setText("report-total", summary.total ?? records.length);
    setText("report-danger", significant);
    setText(
        "report-cost-sum",
        summary.unpriced_count > 0
            ? `전체 산출 보류 (${summary.unpriced_count}건 미산출)` +
              (summary.priced_subtotal_krw != null
                ? ` · 산출된 ${summary.priced_count}건만 ${formatCost(summary.priced_subtotal_krw)}` : "")
            : (summary.estimated_cost_krw != null ? formatCost(summary.estimated_cost_krw) : "기록 없음")
    );

    if (summary.depth_scenarios?.length) {
        $("report-cost-sum").innerHTML = depthCostTable(summary.depth_scenarios) +
            '<p class="stat-note">각 행은 모든 주입 보수 균열이 해당 깊이라고 가정한 합계입니다. 표면처리는 동일 비용으로 포함합니다. 이전 단일 깊이 기록은 비교 합계에서 제외하며 미산출로 표시합니다.</p>';
    }

    const toList = (target, entries, suffix) => {
        const element = $(target);
        const keys = Object.keys(entries || {});
        if (keys.length === 0) {
            element.innerHTML = '<li class="stat-note">기록 없음</li>';
            return;
        }
        element.innerHTML = keys.sort().map((key) => `
            <li>${escapeHtml(key)}${suffix}: 손상 ${entries[key]}건</li>
        `).join("");
    };

    toList("summary-member", summary.by_member, "");
    toList("summary-grade", summary.by_grade, "등급");
}

function sizeText(crack) {
    const parts = [];
    if (crack.max_width_mm) {
        parts.push("폭(P95) " + formatNumber(crack.max_width_mm, 2, " mm"));
    }
    if (crack.length_mm) {
        parts.push("길이 " + formatNumber(crack.length_mm, 1, " mm"));
    }
    if (crack.area_mm2) {
        parts.push("면적 " + formatNumber(crack.area_mm2 / 1e6, 3, " ㎡"));
    }
    return parts.join(" · ") || "-";
}

function filterRecords(records) {
    return dangerOnly
        ? records.filter((crack) => crack.grade && "cde".includes(crack.grade))
        : records;
}

function renderDamageTable(records) {
    const rows = filterRecords(records);
    const tbody = $("report-table-body");

    if (rows.length === 0) {
        tbody.innerHTML = `<tr class="empty-row"><td colspan="8">${
            dangerOnly ? "c등급 이상 손상 기록이 없습니다." : "기록된 손상이 없습니다."
        }</td></tr>`;
        return;
    }

    tbody.innerHTML = rows.map((crack) => `
        <tr>
            <td>${escapeHtml(crack.damage_no ?? "-")}</td>
            <td>${escapeHtml(crack.member ?? "-")}</td>
            <td>${escapeHtml(crack.position ?? "-")}</td>
            <td>${escapeHtml(crack.defect ?? "균열")}</td>
            <td>${escapeHtml(sizeText(crack))}</td>
            <td class="${riskClass(crack.risk_code)}">
                ${escapeHtml(crack.grade ? crack.grade + "등급 · " + crack.preliminary_grade : "판정 보류")}
            </td>
            <td>${escapeHtml(crack.action ?? "-")}</td>
            <td>${escapeHtml(crack.timestamp)}</td>
        </tr>
    `).join("");
}

function depthCostTable(rows) {
    return '<table><thead><tr><th>가정한 내부 깊이</th><th>예상 비용</th></tr></thead><tbody>' + rows.map(row => {
        const label = row.assumed_depth_cm == null ? "표면처리 · 깊이 무관" : `${row.assumed_depth_cm}cm일 경우`;
        const amount = row.unpriced_count > 0
            ? `전체 산출 보류 (${row.unpriced_count}건 미산출)` + (row.priced_subtotal_krw != null ? ` · 산출된 건만 ${formatCost(row.priced_subtotal_krw)}` : "")
            : (row.estimated_cost_krw != null ? `약 ${formatCost(row.estimated_cost_krw)}` : "산출 보류");
        return `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(amount)}</td></tr>`;
    }).join("") + '</tbody></table><p class="stat-note">내부 깊이는 미측정입니다. 가정별 비용을 서로 합산하지 않습니다.</p>';
}

function widthDetails(crack) {
    if (!crack.width_profile_version) return '<p class="stat-note">이전 기록: 최대 틈폭·변화 이력 없음</p>';
    const signed = value => value == null ? "비교 보류" : `${value > 0 ? "+" : ""}${Number(value).toFixed(3)} mm`;
    const baseline = crack.width_baseline;
    const difference = crack.raw_max_width_mm != null && crack.mean_width_mm != null
        ? crack.raw_max_width_mm - crack.mean_width_mm : null;
    const history = [...(crack.width_history || [])].reverse();
    return `<h4>틈폭 관측</h4><p>최대 틈폭 ${formatNumber(crack.raw_max_width_mm, 3, " mm")} · 대표 폭(중앙값) ${formatNumber(crack.mean_width_mm, 3, " mm")} · 두 폭의 차이 ${formatNumber(difference, 3, " mm")}</p>
        <p>최초 비교 기준 ${baseline ? escapeHtml(baseline.timestamp) + " / " + formatNumber(baseline.width_p95_mm, 3, " mm") : "대기"} · 최초 대비 P95 변화 ${signed(crack.width_delta_mm)} · 직전 대비 ${signed(crack.width_previous_delta_mm)}</p>
        <p class="stat-note">${escapeHtml(crack.width_change_status || "첫 이력 저장 대기")} · 촬영·분할 오차를 포함한 관측값 차이이며 성장 확정값이 아닙니다.</p>
        <details><summary>폭 관측 이력 (${crack.width_sample_count || 0}회 중 최근 ${history.length}회)</summary>
        <table><thead><tr><th>시간</th><th>최대 틈폭</th><th>P95</th><th>최초 대비</th><th>예비 구간</th></tr></thead><tbody>${history.map(sample => `<tr>
            <td>${escapeHtml(sample.timestamp)}</td><td>${formatNumber(sample.raw_max_width_mm, 3, " mm")}</td>
            <td>${formatNumber(sample.width_p95_mm, 3, " mm")}</td><td>${signed(sample.width_delta_mm)}</td>
            <td>${escapeHtml(sample.grade || "보류")}</td></tr>`).join("")}</tbody></table></details>`;
}

function costDetails(crack) {
    const basis = crack.cost_basis;
    const breakdown = crack.cost_breakdown;
    let text = `<p class="stat-note">${escapeHtml(crack.cost_status ?? "산출 근거 미확인")}</p>`;
    if (crack.cost_scenarios?.length) {
        const values = basis?.scenario_inputs || {};
        return text + depthCostTable(crack.cost_scenarios) +
            `<p>폭 기준: ${values.width_mode === "assumed" ? "가정 폭" : "관측 대표 폭"} ${escapeHtml(basis?.width_used_mm ?? "-")}mm · 메모: ${escapeHtml(values.note || "없음")}</p>` +
            `<p class="stat-note">${escapeHtml(basis?.standard?.title)} / ${escapeHtml(basis?.method?.section)} · 제외: ${escapeHtml((crack.cost_exclusions || []).join(", "))}</p>`;
    }
    if (!basis || !breakdown) return text;
    if (basis.scenario_inputs) {
        const values = basis.scenario_inputs;
        const width = values.width_mode === "assumed" ? "가정 폭" : "관측 대표 폭";
        text += `<p>${escapeHtml(basis.method?.label)} (가정) · ${values.method === "injection" ? `내부 깊이 ${escapeHtml(values.depth_mm)}mm 가정 · ${width} ${escapeHtml(basis.width_used_mm)}mm · ` : ""}재료 사용량 ${formatNumber(basis.material_kg_per_m, 4, " kg/m")}</p>`;
        text += `<p>설정 메모: ${escapeHtml(values.note || "없음")} · 저장일: ${escapeHtml(basis.saved_at)}</p>`;
    }
    const labels = {labor: "노무비", tools: "공구·경장비", material: "주재료비", misc_material: "잡재료비"};
    text += `<p>${Object.entries(labels).map(([key, label]) =>
        `${label}: ${escapeHtml(formatCost(breakdown[key]))}`).join(" · ")}</p>`;
    const sources = [
        ...Object.values(basis.labor_rates || {}).map(item => item.source),
        basis.profile?.material?.quantity_source,
        basis.profile?.material?.price_source,
        basis.profile?.misc_material_source,
    ].filter(Boolean);
    text += `<p>적용 근거: ${escapeHtml(basis.standard?.title)} / ${escapeHtml(basis.method?.section)} / ${escapeHtml(basis.standard?.pages)}</p>`;
    if (basis.review_note) text += `<p>공법 검토: ${escapeHtml(basis.review_note)}</p>`;
    text += `<ul>${sources.map(source => `<li>${escapeHtml(source.title)} · ${escapeHtml(source.date)} · ${escapeHtml(source.reference)}</li>`).join("")}</ul>`;
    text += `<p class="stat-note">제외: ${escapeHtml((crack.cost_exclusions || []).join(", "))}</p>`;
    return text;
}

function renderDamageDetails(records) {
    const container = $("damage-detail-list");
    const rows = filterRecords(records).slice(0, 20);

    if (rows.length === 0) {
        container.innerHTML = '<p class="stat-note">기록된 손상이 없습니다.</p>';
        return;
    }

    container.innerHTML = rows.map((crack) => `
        <article class="damage-detail">
            ${crack.image_url
                ? `<img src="${escapeHtml(crack.image_url)}" alt="${escapeHtml(crack.damage_no ?? "손상")} 탐지 화면" loading="lazy">`
                : '<div class="placeholder">저장된 이미지 없음</div>'}
            <div>
                <h4>${escapeHtml(crack.damage_no ?? "-")} · ${escapeHtml(crack.defect ?? "균열")}</h4>
                <dl>
                    <dt>부재</dt><dd>${escapeHtml(crack.member ?? "-")} (${escapeHtml(crack.member_material_label ?? "-")})</dd>
                    <dt>위치</dt><dd>${escapeHtml(crack.position ?? "-")}</dd>
                    <dt>AI 신뢰도</dt><dd>${formatNumber(crack.confidence * 100, 1, "%")}</dd>
                    <dt>추정 크기</dt><dd>${escapeHtml(sizeText(crack))}</dd>
                    <dt>깊이(단차)</dt><dd>${
                        crack.depth_diff_mm != null
                            ? formatNumber(crack.depth_diff_mm, 1, " mm") + ` (${escapeHtml(crack.depth_status)})`
                            : escapeHtml(crack.depth_status ?? "측정 불가")
                    }</dd>
                    <dt>치수 산출</dt><dd>${escapeHtml(scaleLabel(crack))}</dd>
                    <dt>판정 근거</dt><dd>${escapeHtml(crack.grade_basis ?? "-")}</dd>
                    <dt>예비 상태</dt><dd class="${riskClass(crack.risk_code)}">${escapeHtml(crack.risk_level)}</dd>
                    <dt>권고 조치</dt><dd>${escapeHtml(crack.action ?? "-")}</dd>
                    <dt>선택 공법 / 직접작업비</dt><dd>${escapeHtml(crack.recommended_repair_method ?? "-")} · ${escapeHtml(formatCost(crack.estimated_repair_cost_krw))}</dd>
                    <dt>발견 시간</dt><dd>${escapeHtml(crack.timestamp)}</dd>
                </dl>
                ${widthDetails(crack)}
                ${costDetails(crack)}
            </div>
        </article>
    `).join("");
}

// ----------------------------------------------------
// 점검 기본정보
// ----------------------------------------------------

const INSPECTION_FIELDS = [
    "bridge_name", "bridge_id", "site", "inspector", "weather",
    "capture_distance_mm", "inspection_scope", "member_group", "member",
    "member_label", "member_material", "member_position",
];

let MEMBER_GROUPS = {};
let MATERIAL_LABELS = {};

const GRADE_HINTS = {
    rc: "0.1 / 0.3 / 0.5 / 1.0 mm",
    psc: "0.2 / 0.3 / 0.5 mm",
    rc_crossbeam: "0.1 / 0.3 / 0.5 mm",
    steel: "균열폭 기준 없음",
};

function fillSelect(select, values, selected) {
    select.innerHTML = values.map(([value, label]) =>
        `<option value="${escapeHtml(value)}"${value === selected ? " selected" : ""}>${escapeHtml(label)}</option>`
    ).join("");
}

function applyInspection(info) {
    fillSelect(
        $("f-member_group"),
        Object.keys(MEMBER_GROUPS).map((key) => [key, key]),
        info.member_group
    );
    fillSelect(
        $("f-member"),
        (MEMBER_GROUPS[info.member_group] || []).map((name) => [name, name]),
        info.member
    );
    fillSelect(
        $("f-member_material"),
        Object.entries(MATERIAL_LABELS),
        info.member_material
    );

    INSPECTION_FIELDS.forEach((key) => {
        const field = $("f-" + key);
        if (field && !["member_group", "member", "member_material"].includes(key)) {
            field.value = info[key] ?? "";
        }
    });

    setText(
        "current-member",
        [info.member_label || info.member, MATERIAL_LABELS[info.member_material]]
            .filter(Boolean).join(" · ")
    );
    setText("grade-hint", GRADE_HINTS[info.member_material] ?? "-");
}

async function loadInspection() {
    try {
        const response = await fetch("/api/inspection", { cache: "no-store" });
        const payload = await response.json();
        MEMBER_GROUPS = payload.member_groups || {};
        MATERIAL_LABELS = payload.materials || {};
        applyInspection(payload.inspection || {});
    } catch (error) {
        console.error("점검 기본정보 조회 오류:", error);
    }
}

async function saveInspection(button) {
    const values = {};
    INSPECTION_FIELDS.forEach((key) => {
        const field = $("f-" + key);
        if (!field) {
            return;
        }
        values[key] = key === "capture_distance_mm"
            ? (field.value === "" ? null : Number(field.value))
            : field.value;
    });

    button.disabled = true;
    try {
        const response = await fetch("/api/inspection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(values),
        });
        const payload = await response.json();

        if (payload.success) {
            applyInspection(payload.inspection);
            showToast("기본정보를 저장했습니다.");
            loadReport();
        } else {
            showToast(payload.message ?? "저장에 실패했습니다.", true);
        }
    } catch (error) {
        console.error(error);
        showToast("기본정보를 저장하지 못했습니다.", true);
    } finally {
        button.disabled = false;
    }
}

async function loadCaptures() {
    const container = $("capture-list");
    try {
        const response = await fetch("/api/captures", { cache: "no-store" });
        const payload = await response.json();
        const captures = payload.captures || [];

        if (captures.length === 0) {
            container.innerHTML = '<p class="stat-note">저장된 캡처가 없습니다.</p>';
            return;
        }

        container.innerHTML = captures.slice(0, 24).map((item) => `
            <div class="capture-item">
                ${item.image_url
                    ? `<img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.name)} 캡처" loading="lazy">`
                    : '<div class="placeholder">이미지 없음</div>'}
                <div>${escapeHtml(item.captured_at ?? item.name)}</div>
                <a href="${escapeHtml(item.json_url)}" target="_blank" rel="noopener">측정값 JSON</a>
            </div>
        `).join("");
    } catch (error) {
        console.error("캡처 목록 오류:", error);
        container.innerHTML = '<p class="stat-note">캡처 목록을 불러오지 못했습니다.</p>';
    }
}

// ----------------------------------------------------
// 동작
// ----------------------------------------------------

async function captureCurrentFrame(button) {
    button.disabled = true;
    try {
        const response = await fetch("/api/capture", { method: "POST" });
        const payload = await response.json();

        if (payload.success) {
            showToast(`저장했습니다: ${payload.name}`);
            if (currentPage === "report") {
                loadCaptures();
            }
        } else {
            showToast(payload.message ?? "저장에 실패했습니다.", true);
        }
    } catch (error) {
        console.error(error);
        showToast("저장 요청을 보내지 못했습니다. 서버 연결을 확인하세요.", true);
    } finally {
        button.disabled = false;
    }
}

async function clearHistory() {
    if (!confirm("저장된 균열 이력을 모두 삭제할까요? 캡처 파일은 유지됩니다. 관측 중이면 새 기록이 다시 추가됩니다.")) {
        return;
    }
    try {
        const response = await fetch("/api/history", { method: "DELETE" });
        if (!response.ok) {
            throw new Error("HTTP " + response.status);
        }
        showToast("이력을 지웠습니다.");
        loadHistory();
    } catch (error) {
        console.error(error);
        showToast("이력을 지우지 못했습니다.", true);
    }
}

function printReport() {
    const report = $("report");
    report.classList.add("print-target");
    window.print();
    setTimeout(() => report.classList.remove("print-target"), 500);
}

document.addEventListener("click", (event) => {
    const nav = event.target.closest(".nav-item");
    if (nav) {
        switchPage(nav.dataset.page);
        return;
    }

    const tab = event.target.closest(".tab");
    if (tab) {
        switchTab(tab.dataset.tab);
        return;
    }

    const button = event.target.closest("[data-action]");
    if (!button) {
        return;
    }

    const actions = {
        capture: () => captureCurrentFrame(button),
        print: printReport,
        "refresh-report": loadReport,
        "clear-history": clearHistory,
        "save-inspection": () => saveInspection(button),
        "estimate-setup": () => loadEstimateSetup(true),
    };
    actions[button.dataset.action]?.();
});

$("danger-only").addEventListener("change", (event) => {
    dangerOnly = event.target.checked;
    loadHistory();
});

$("f-member_group").addEventListener("change", (event) => {
    const members = MEMBER_GROUPS[event.target.value] || [];
    fillSelect($("f-member"), members.map((name) => [name, name]), members[0]);
});

// 탭이 백그라운드로 가면 스트림과 폴링을 멈춰 로봇 PC 부하를 줄입니다.
document.addEventListener("visibilitychange", () => {
    syncStreams();

    if (document.hidden) {
        clearTimeout(pollTimer);
        clearTimeout(reportTimer);
    } else {
        clearTimeout(pollTimer);
        pollLatest();
        scheduleReport(currentPage === "report");
    }
});

syncStreams();
loadInspection();
loadEstimateSetup();
pollLatest();
