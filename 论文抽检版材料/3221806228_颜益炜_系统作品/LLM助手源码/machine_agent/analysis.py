"""Explainable condition-summary and abnormal-pattern analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from machine_agent.models import AlarmRecord, MachineHistoryPoint, TimeRange


def summarize_machine_condition(
    history: list[MachineHistoryPoint],
    alarms: list[AlarmRecord],
    time_range: TimeRange,
) -> dict[str, Any]:
    """Summarize operation, telemetry, output, and alarms for a time range."""
    if not history:
        return {
            "time_range": time_range.to_dict(),
            "sample_count": 0,
            "summary_text": f"{time_range.label}没有可用历史数据，无法判断设备运行情况。",
        }

    running_points = [point for point in history if point.state in {"running", "warning"}]
    warning_points = [point for point in history if point.state == "warning"]
    output_delta = max(point.output_count for point in history) - min(point.output_count for point in history)
    alarm_counter = Counter(alarm.alarm_type for alarm in alarms)
    top_alarm = alarm_counter.most_common(1)[0] if alarm_counter else ("none", 0)
    avg_temperature = mean(point.temperature_c for point in history)
    avg_pressure = mean(point.pressure_mpa for point in history)
    avg_current = mean(point.current_a for point in history)
    max_temperature = max(point.temperature_c for point in history)
    max_current = max(point.current_a for point in history)
    utilization = len(running_points) / len(history) * 100.0

    summary_text = (
        f"{time_range.label}共读取 {len(history)} 条设备历史记录，运行/告警采样占比约 {utilization:.1f}%。"
        f"平均温度 {avg_temperature:.1f}C，最高温度 {max_temperature:.1f}C；"
        f"平均压力 {avg_pressure:.3f}MPa，平均电流 {avg_current:.1f}A，最高电流 {max_current:.1f}A。"
        f"区间产量增加 {output_delta} 件。"
    )
    if alarms:
        summary_text += (
            f" 同期出现 {len(alarms)} 次报警，最频繁类型是 {top_alarm[0]}（{top_alarm[1]} 次）。"
        )
    else:
        summary_text += " 同期未记录报警。"
    if warning_points:
        summary_text += f" 有 {len(warning_points)} 个采样点处于 warning，建议优先核查温度、电流和冷却状态。"

    return {
        "time_range": time_range.to_dict(),
        "sample_count": len(history),
        "alarm_count": len(alarms),
        "running_sample_count": len(running_points),
        "warning_sample_count": len(warning_points),
        "utilization_pct": round(utilization, 2),
        "output_delta": output_delta,
        "avg_temperature_c": round(avg_temperature, 2),
        "max_temperature_c": round(max_temperature, 2),
        "avg_pressure_mpa": round(avg_pressure, 3),
        "avg_current_a": round(avg_current, 2),
        "max_current_a": round(max_current, 2),
        "top_alarm_type": {"type": top_alarm[0], "count": top_alarm[1]},
        "alarm_type_counts": dict(alarm_counter),
        "summary_text": summary_text,
    }


def detect_abnormal_patterns(
    history: list[MachineHistoryPoint],
    alarms: list[AlarmRecord],
    time_range: TimeRange,
) -> dict[str, Any]:
    """Find high-frequency alarms, high-risk hours, and telemetry differences."""
    alarm_counter = Counter(alarm.alarm_type for alarm in alarms)
    hourly_counter = Counter(alarm.timestamp.hour for alarm in alarms)
    top_alarm_types = [
        {"type": alarm_type, "count": count}
        for alarm_type, count in alarm_counter.most_common(5)
    ]
    high_risk_hours = [
        {"hour": hour, "count": count}
        for hour, count in hourly_counter.most_common(3)
    ]

    normal_points: list[MachineHistoryPoint] = []
    abnormal_points: list[MachineHistoryPoint] = []
    for point in history:
        near_alarm = any(
            abs((point.timestamp - alarm.timestamp).total_seconds()) <= 6 * 3600
            for alarm in alarms
        )
        if point.state == "warning" or point.temperature_c >= 82.0 or point.current_a >= 36.0 or near_alarm:
            abnormal_points.append(point)
        else:
            normal_points.append(point)

    differences = _metric_differences(normal_points, abnormal_points)
    root_causes = _infer_root_causes(alarm_counter, differences)
    conclusion = _build_pattern_summary(
        time_range=time_range,
        alarm_count=len(alarms),
        top_alarm_types=top_alarm_types,
        high_risk_hours=high_risk_hours,
        differences=differences,
        root_causes=root_causes,
    )

    return {
        "time_range": time_range.to_dict(),
        "alarm_count": len(alarms),
        "top_alarm_types": top_alarm_types,
        "high_risk_hours": high_risk_hours,
        "normal_sample_count": len(normal_points),
        "abnormal_sample_count": len(abnormal_points),
        "metric_differences": differences,
        "possible_causes": root_causes,
        "summary_text": conclusion,
    }


def _metric_differences(
    normal_points: list[MachineHistoryPoint],
    abnormal_points: list[MachineHistoryPoint],
) -> dict[str, Any]:
    metrics = ("temperature_c", "pressure_mpa", "rpm", "current_a")
    if not abnormal_points:
        return {}
    if not normal_points:
        normal_points = abnormal_points
    result: dict[str, Any] = {}
    for metric in metrics:
        normal_avg = mean(float(getattr(point, metric)) for point in normal_points)
        abnormal_avg = mean(float(getattr(point, metric)) for point in abnormal_points)
        result[metric] = {
            "normal_avg": round(normal_avg, 3),
            "abnormal_avg": round(abnormal_avg, 3),
            "delta": round(abnormal_avg - normal_avg, 3),
        }
    return result


def _infer_root_causes(
    alarm_counter: Counter[str],
    differences: dict[str, Any],
) -> list[str]:
    causes: list[str] = []
    if alarm_counter.get("temperature_high", 0) >= 2:
        causes.append("温度报警集中出现，优先检查冷却水路、风扇、滤网堵塞和连续高负载。")
    if alarm_counter.get("pressure_drop", 0) >= 1:
        causes.append("存在压力下跌报警，建议检查液压油位、滤芯和泄压阀。")
    if alarm_counter.get("over_current", 0) >= 1:
        causes.append("存在过流报警，建议检查电机负载、轴承阻力和传动机构卡滞。")
    temp_delta = float(differences.get("temperature_c", {}).get("delta", 0.0))
    current_delta = float(differences.get("current_a", {}).get("delta", 0.0))
    if temp_delta > 3.0 and current_delta > 1.0:
        causes.append("异常时温度和电流同时升高，更像负载升高或散热效率下降，而不是单一传感器误报。")
    if not causes:
        causes.append("未发现单一高频异常类型，建议继续观察趋势并核查传感器采样质量。")
    return causes


def _build_pattern_summary(
    *,
    time_range: TimeRange,
    alarm_count: int,
    top_alarm_types: list[dict[str, Any]],
    high_risk_hours: list[dict[str, Any]],
    differences: dict[str, Any],
    root_causes: list[str],
) -> str:
    if alarm_count == 0:
        return f"{time_range.label}未发现报警记录，规则分析没有识别出明显异常模式。"
    top_alarm = top_alarm_types[0] if top_alarm_types else {"type": "unknown", "count": 0}
    hour_text = "、".join(f"{item['hour']}点" for item in high_risk_hours) or "未形成固定时段"
    temp_delta = differences.get("temperature_c", {}).get("delta")
    current_delta = differences.get("current_a", {}).get("delta")
    comparison = ""
    if temp_delta is not None and current_delta is not None:
        comparison = f" 异常采样相对正常采样温度变化 {temp_delta:+.1f}C，电流变化 {current_delta:+.1f}A。"
    return (
        f"{time_range.label}共发现 {alarm_count} 次报警，最高频类型为 {top_alarm['type']}（{top_alarm['count']} 次），"
        f"高发时段集中在 {hour_text}。{comparison}"
        f" 可能原因：{' '.join(root_causes)}"
    )


def group_alarms_by_status(alarms: list[AlarmRecord]) -> dict[str, int]:
    """Return status distribution for dashboards."""
    grouped: defaultdict[str, int] = defaultdict(int)
    for alarm in alarms:
        grouped[alarm.status] += 1
    return dict(grouped)
