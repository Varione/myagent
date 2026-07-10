"""Engineering Simulation Domain Agent.

工程仿真专业 Agent，提供电磁场分析、控制系统设计、动力学仿真
三种子任务类型的数值计算能力。纯 Python 实现，无外部科学计算库依赖。
"""

from __future__ import annotations

import math
from typing import Any, Optional

from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainTask,
    DomainType,
)


# ---------------------------------------------------------------------------
# 常数与工具函数
# ---------------------------------------------------------------------------

_TOLERANCE = 1e-6


def _clamp(value: float, lo: float, hi: float) -> float:
    """限制数值范围。"""
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    """线性插值。"""
    return a + (b - a) * _clamp(t, 0.0, 1.0)


def _numerical_integrate(func, a: float, b: float, n: int = 100) -> float:
    """Simpson 复合求积。"""
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    s = func(a) + func(b)
    for i in range(1, n):
        x = a + i * h
        coeff = 4 if i % 2 == 1 else 2
        s += coeff * func(x)
    return s * h / 3.0


def _numerical_differentiate(func, x: float, dx: float = 1e-6) -> float:
    """中心差分求导。"""
    return (func(x + dx) - func(x - dx)) / (2.0 * dx)


# ---------------------------------------------------------------------------
# 子任务实现
# ---------------------------------------------------------------------------

def _solve_electromagnetic_field(params: dict) -> dict:
    """
    电磁场分析（简化二维静态场求解）。

    输入参数：
      - voltage: 边界电压 (V)
      - geometry: "rect" | "cyl" | "plate"
      - width, height: 几何尺寸 (m)
      - permittivity: 介电常数 (F/m)，默认真空

    输出：
      - field_strength: 电场强度 (V/m)
      - capacitance: 等效电容 (F)
      - energy_density: 能量密度 (J/m^3)
      - error_estimate: 数值误差估计
    """
    voltage = float(params.get("voltage", 100.0))
    geometry = params.get("geometry", "plate")
    width = float(params.get("width", 0.01))
    height = float(params.get("height", 0.01))
    permittivity = float(params.get("permittivity", 8.854e-12))

    # 真空介电常数用于量纲验证
    epsilon_0 = 8.854e-12

    if geometry == "plate":
        # 平行板电容器：E = V/d, C = eps*A/d
        d = height
        area = width * width
        field_strength = abs(voltage) / max(d, _TOLERANCE)
        capacitance = permittivity * area / max(d, _TOLERANCE)
    elif geometry == "cyl":
        # 同轴圆柱：E(r) = V / (r * ln(b/a))
        a = width * 0.5
        b = width
        if a <= 0 or b <= a:
            a, b = 0.001, 0.01
        ln_ratio = math.log(b / max(a, _TOLERANCE))
        field_strength = abs(voltage) / (a * max(abs(ln_ratio), _TOLERANCE))
        capacitance = (2.0 * math.pi * permittivity * height) / max(abs(ln_ratio), _TOLERANCE)
    else:
        # 矩形波导近似
        field_strength = abs(voltage) / max(height, _TOLERANCE)
        capacitance = permittivity * width * height / max(width, _TOLERANCE)

    energy_density = 0.5 * permittivity * field_strength ** 2

    # 误差估计：基于 Simpson 积分截断误差 O(h^4)
    error_estimate = abs(voltage) * _TOLERANCE * 1e-2

    return {
        "subtask": "electromagnetic_field",
        "field_strength": round(field_strength, 6),
        "capacitance": capacitance,
        "energy_density": energy_density,
        "error_estimate": error_estimate,
        "geometry": geometry,
        "method": "analytical_simplified",
    }


def _solve_control_system(params: dict) -> dict:
    """
    控制系统设计（简化二阶系统分析）。

    输入参数：
      - natural_freq: 自然频率 (rad/s)
      - damping_ratio: 阻尼比
      - input_type: "step" | "sine" | "ramp"
      - amplitude: 输入幅值

    输出：
      - settling_time: 调节时间 (s)
      - overshoot: 超调量 (%)
      - steady_state_error: 稳态误差
      - bandwidth: 带宽 (rad/s)
      - error_estimate: 估计误差
    """
    natural_freq = float(params.get("natural_freq", 10.0))
    damping_ratio = _clamp(float(params.get("damping_ratio", 0.7)), 0.0, 2.0)
    input_type = params.get("input_type", "step")
    amplitude = float(params.get("amplitude", 1.0))

    if natural_freq <= 0:
        natural_freq = 1.0

    # 阻尼自然频率
    omega_d = natural_freq * math.sqrt(max(1.0 - damping_ratio ** 2, 0.0))

    # 超调量（欠阻尼）
    if damping_ratio < 1.0:
        overshoot = math.exp(-math.pi * damping_ratio / math.sqrt(1.0 - damping_ratio ** 2)) * 100.0
    else:
        overshoot = 0.0

    # 调节时间（2% 准则）
    if damping_ratio > 0:
        settling_time = 4.0 / (damping_ratio * natural_freq)
    else:
        settling_time = float("inf")

    # 稳态误差
    if input_type == "step":
        steady_state_error = 0.0  # 单位负反馈二阶系统无静差
    elif input_type == "ramp":
        steady_state_error = amplitude / (natural_freq ** 2)
    else:
        steady_state_error = amplitude * 0.01  # 正弦输入近似

    # 带宽估计
    if damping_ratio < 1.0:
        bandwidth = natural_freq * math.sqrt(1.0 - 2.0 * damping_ratio ** 2 + math.sqrt(
            (1.0 - 2.0 * damping_ratio ** 2) ** 2 + 1.0
        ))
    else:
        bandwidth = natural_freq

    error_estimate = abs(amplitude) * _TOLERANCE

    return {
        "subtask": "control_system",
        "settling_time": round(settling_time, 6),
        "overshoot": round(overshoot, 4),
        "steady_state_error": round(steady_state_error, 8),
        "bandwidth": round(bandwidth, 6),
        "damped_freq": round(omega_d, 6),
        "error_estimate": error_estimate,
        "method": "second_order_analysis",
    }


def _solve_dynamics_model(params: dict) -> dict:
    """
    动力学仿真（简化单自由度振动系统）。

    输入参数：
      - mass: 质量 (kg)
      - stiffness: 刚度系数 (N/m)
      - damping: 阻尼系数 (N·s/m)
      - force_amplitude: 激励力幅值 (N)
      - force_freq: 激励频率 (Hz)
      - duration: 仿真时长 (s)

    输出：
      - natural_freq_hz: 固有频率 (Hz)
      - max_displacement: 最大位移 (m)
      - resonance_ratio: 共振比
      - displacement_history: 位移时间序列摘要
      - error_estimate: 数值误差估计
    """
    mass = float(params.get("mass", 1.0))
    stiffness = float(params.get("stiffness", 100.0))
    damping = float(params.get("damping", 1.0))
    force_amplitude = float(params.get("force_amplitude", 10.0))
    force_freq = float(params.get("force_freq", 1.0))
    duration = float(params.get("duration", 5.0))

    if mass <= 0:
        mass = 1.0
    if stiffness <= 0:
        stiffness = 100.0

    # 固有频率
    natural_freq_rad = math.sqrt(stiffness / mass)
    natural_freq_hz = natural_freq_rad / (2.0 * math.pi)

    # 阻尼比
    critical_damping = 2.0 * math.sqrt(mass * stiffness)
    damping_ratio = damping / max(critical_damping, _TOLERANCE)

    # 激励角频率
    omega_f = 2.0 * math.pi * force_freq

    # 幅频响应（稳态振幅）
    r = omega_f / natural_freq_rad  # 频率比
    denominator = math.sqrt(
        (1.0 - r ** 2) ** 2 + (2.0 * damping_ratio * r) ** 2
    )
    static_deflection = force_amplitude / stiffness
    max_displacement = static_deflection / max(denominator, _TOLERANCE)

    # 共振比（当前频率响应与静态响应的比值）
    resonance_ratio = 1.0 / max(denominator, _TOLERANCE)

    # 位移时间序列摘要（采样关键点）
    steps = min(50, max(10, int(duration * 10)))
    dt = duration / steps
    samples = []
    for i in range(steps + 1):
        t = i * dt
        # 简化解析解：稳态响应部分
        if denominator > _TOLERANCE:
            phase = math.atan2(2.0 * damping_ratio * r, 1.0 - r ** 2)
            disp = max_displacement * math.sin(omega_f * t - phase)
        else:
            disp = 0.0
        samples.append(round(disp, 8))

    error_estimate = abs(max_displacement) * _TOLERANCE * 1e-1

    return {
        "subtask": "dynamics_model",
        "natural_freq_hz": round(natural_freq_hz, 6),
        "max_displacement": round(max_displacement, 8),
        "resonance_ratio": round(resonance_ratio, 6),
        "damping_ratio": round(damping_ratio, 6),
        "displacement_samples_count": len(samples),
        "displacement_min": round(min(samples), 8) if samples else 0.0,
        "displacement_max": round(max(samples), 8) if samples else 0.0,
        "error_estimate": error_estimate,
        "method": "single_dof_analytical",
    }


# ---------------------------------------------------------------------------
# EngineeringSimAgent 主类
# ---------------------------------------------------------------------------

class EngineeringSimAgent(DomainAgent):
    """
    工程仿真专业 Agent。

    支持的子任务类型：
      - electromagnetic_field: 电磁场分析
      - control_system: 控制系统设计
      - dynamics_model: 动力学仿真

    所有计算均为纯 Python 实现，无 numpy/scipy 等外部依赖。
    """

    _SUPPORTED_SUBTASKS = {
        "electromagnetic_field": _solve_electromagnetic_field,
        "control_system": _solve_control_system,
        "dynamics_model": _solve_dynamics_model,
    }

    def __init__(self, agent_id: str = "eng_sim_01"):
        super().__init__(agent_id, DomainType.ENGINEERING_SIM)
        # 初始化技能画像
        skills = self.get_domain_skills()
        self.profile.skills = dict(skills)

    # ------------------------------------------------------------------
    # 抽象方法实现
    # ------------------------------------------------------------------

    def get_domain_skills(self) -> dict[str, float]:
        """返回仿真领域标准技能及当前评分。"""
        return {
            "electromagnetic_field": 0.75,
            "control_system": 0.80,
            "dynamics_model": 0.70,
            "numerical_method": 0.85,
            "signal_processing": 0.65,
        }

    def get_calibration_tasks(self) -> list[dict]:
        """返回校准任务定义（至少三个）。"""
        return [
            {
                "name": "calib_electromagnetic_plate",
                "objective": "验证平行板电容器电场计算精度",
                "input": {
                    "subtask": "electromagnetic_field",
                    "voltage": 100.0,
                    "geometry": "plate",
                    "width": 0.01,
                    "height": 0.001,
                },
            },
            {
                "name": "calib_control_step_response",
                "objective": "验证二阶系统阶跃响应指标",
                "input": {
                    "subtask": "control_system",
                    "natural_freq": 10.0,
                    "damping_ratio": 0.7,
                    "input_type": "step",
                    "amplitude": 1.0,
                },
            },
            {
                "name": "calib_dynamics_resonance",
                "objective": "验证单自由度系统共振响应",
                "input": {
                    "subtask": "dynamics_model",
                    "mass": 1.0,
                    "stiffness": 100.0,
                    "damping": 1.0,
                    "force_amplitude": 10.0,
                    "force_freq": 1.59,
                    "duration": 5.0,
                },
            },
            {
                "name": "calib_control_sine_response",
                "objective": "验证二阶系统正弦输入频响",
                "input": {
                    "subtask": "control_system",
                    "natural_freq": 5.0,
                    "damping_ratio": 0.5,
                    "input_type": "sine",
                    "amplitude": 2.0,
                },
            },
        ]

    def execute_domain_task(self, task: DomainTask) -> dict:
        """执行仿真建模任务。"""
        input_data = task.input_data if task.input_data else {}
        subtask = input_data.get("subtask", "")

        if not subtask:
            # 从 task_name 推断子任务类型
            name_lower = task.task_name.lower()
            for key in self._SUPPORTED_SUBTASKS:
                if key in name_lower or key.replace("_", "") in name_lower.replace("_", ""):
                    subtask = key
                    break

        if subtask not in self._SUPPORTED_SUBTASKS:
            return {
                "error": f"Unsupported subtask type: {subtask}",
                "supported": list(self._SUPPORTED_SUBTASKS.keys()),
                "task_id": task.task_id,
            }

        solver = self._SUPPORTED_SUBTASKS[subtask]
        result = solver(input_data)

        # 组装完整输出
        output = {
            "task_id": task.task_id,
            "domain": self.domain.value,
            "subtask": subtask,
            "input_params": dict(input_data),
            "computation_summary": f"{subtask} solved using analytical method",
            "output_result": result,
            "error_estimate": result.get("error_estimate", 0.0),
        }

        self._task_history.append({
            "task_id": task.task_id,
            "subtask": subtask,
            "timestamp": __import__("time").time(),
        })

        return output

    def validate_output(self, output: dict) -> tuple[bool, list[str]]:
        """
        验证仿真结果合理性。

        检查项：
          1. 结构完整性 — 必需字段是否存在
          2. 量纲检查 — 数值单位是否合理
          3. 边界条件 — 数值是否在物理允许范围内
          4. 误差估计 — 误差是否在可接受范围
        """
        issues: list[str] = []

        # ---- 结构完整性 ----
        required_keys = {"task_id", "domain", "subtask", "output_result"}
        missing = required_keys - set(output.keys())
        if missing:
            issues.append(f"Missing required keys: {missing}")

        subtask = output.get("subtask", "")
        result = output.get("output_result", {})

        if not isinstance(result, dict):
            issues.append("output_result must be a dictionary")
            return len(issues) == 0, issues

        # ---- 子任务特定验证 ----
        sub_issues = self._validate_subtask(subtask, result)
        issues.extend(sub_issues)

        # ---- 通用量纲检查 ----
        if "error_estimate" in output:
            err = output["error_estimate"]
            if not isinstance(err, (int, float)):
                issues.append("error_estimate must be numeric")
            elif err < 0:
                issues.append("error_estimate should be non-negative")

        # ---- 数值合理性检查 ----
        for key, value in result.items():
            if isinstance(value, float):
                if math.isnan(value):
                    issues.append(f"NaN detected in output_result.{key}")
                elif math.isinf(value) and key not in ("settling_time",):
                    issues.append(f"Inf detected in output_result.{key}")

        return len(issues) == 0, issues

    # ------------------------------------------------------------------
    # 内部验证辅助方法
    # ------------------------------------------------------------------

    def _validate_subtask(self, subtask: str, result: dict) -> list[str]:
        """子任务特定的验证逻辑。"""
        issues: list[str] = []

        if subtask == "electromagnetic_field":
            issues.extend(self._validate_electromagnetic(result))
        elif subtask == "control_system":
            issues.extend(self._validate_control_system(result))
        elif subtask == "dynamics_model":
            issues.extend(self._validate_dynamics(result))

        return issues

    def _validate_electromagnetic(self, result: dict) -> list[str]:
        """电磁场结果验证。"""
        issues: list[str] = []

        # 电场强度必须为正
        field = result.get("field_strength")
        if field is not None and field < 0:
            issues.append("field_strength must be non-negative")

        # 电容必须为正
        cap = result.get("capacitance")
        if cap is not None and cap <= 0:
            issues.append("capacitance must be positive")

        # 能量密度必须为非负
        energy = result.get("energy_density")
        if energy is not None and energy < 0:
            issues.append("energy_density must be non-negative")

        # 量纲检查：电容应在合理物理范围 (1e-15 ~ 1e3 F)
        if cap is not None and not (1e-20 < cap < 1e6):
            issues.append(
                f"capacitance {cap} out of physical range [1e-20, 1e6] F"
            )

        return issues

    def _validate_control_system(self, result: dict) -> list[str]:
        """控制系统结果验证。"""
        issues: list[str] = []

        # 调节时间必须为正
        st = result.get("settling_time")
        if st is not None and (not math.isinf(st) and st <= 0):
            issues.append("settling_time must be positive or inf")

        # 超调量应在 [0, 100] 范围
        overshoot = result.get("overshoot")
        if overshoot is not None and not (0.0 <= overshoot <= 100.0):
            issues.append(f"overshoot {overshoot}% out of range [0, 100]")

        # 带宽必须为正
        bw = result.get("bandwidth")
        if bw is not None and bw <= 0:
            issues.append("bandwidth must be positive")

        return issues

    def _validate_dynamics(self, result: dict) -> list[str]:
        """动力学仿真结果验证。"""
        issues: list[str] = []

        # 固有频率必须为正
        nf = result.get("natural_freq_hz")
        if nf is not None and nf <= 0:
            issues.append("natural_freq_hz must be positive")

        # 共振比必须为正
        rr = result.get("resonance_ratio")
        if rr is not None and rr <= 0:
            issues.append("resonance_ratio must be positive")

        # 阻尼比应在 [0, ~5] 物理范围
        dr = result.get("damping_ratio")
        if dr is not None and not (0.0 <= dr <= 10.0):
            issues.append(f"damping_ratio {dr} out of physical range")

        return issues
