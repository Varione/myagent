# DR-MMA: Dynamic Role-based Multi-Model Agent Architecture

## 计划目的

### 要解决的问题

单一大模型在处理复杂任务时存在以下系统性缺陷：

1. **任务拆解不足** — 单一模型难以从多角度全面分解复杂任务，容易出现视角盲区
2. **专业判断偏差** — 单个模型在特定领域的能力有限，缺乏多专家交叉验证
3. **结果校验薄弱** — 生成与审查由同一模型完成，存在自我确认偏差
4. **长流程不可控** — 多步骤执行中缺乏有效的中间检查和动态调整机制
5. **模型数量不固定** — 实际部署中可用模型数量会动态变化，传统固定 agent 架构无法适应

### 核心目标

构建一套**模型数量不固定**条件下可稳定运行的多模型协同 Agent 架构，使系统能够在单模型到多模型集群之间平滑伸缩，同时保持协同推理、任务执行和结果质量控制的能力。

### 设计目标分解

| 目标 | 说明 |
|------|------|
| 多模型协同思考 | 不同模型从规划、执行、批判、校验等角度提出观点，结构化交流 |
| 主控模型统一协调 | Supervisor 负责任务拆解、角色分配、流程调度、冲突裁决和最终汇总 |
| 模型数量不固定 | 模型是资源而非固定 agent，通过角色合并/拆分动态适配 |
| 执行中及时沟通 | agent 通过事件总线实时发出求助、冲突、低置信度等信号 |
| 结果可追踪可审查 | 所有中间过程写入共享黑板和决策日志，支持复盘和回滚 |

---

## 方案

### 核心设计思想：四层解耦

传统多 agent 系统将"模型"与"角色"固定绑定，一旦模型数量变化或不可用，系统面临角色缺失或资源浪费。

本方案采用**四层解耦**：

```
模型 Model      = 可调用的推理资源
能力 Capability  = 模型擅长完成的任务类型画像
角色 Role       = 当前任务中的职责合同（不绑定具体模型）
Agent          = 模型 + 角色 + 工具 + 记忆 + 任务合同（运行时实例）
```

Agent **不是固定实体**，而是运行时生成的执行实例。一个模型可以承担多个角色，一个角色也可以在复杂任务中拆分给多个模型完成。

---

### 系统边界与适用任务

并非所有任务都需要进入多模型协同流程。明确定义系统边界，防止"什么任务都走一遍复杂协同"导致的资源浪费。

#### 适用任务

1. **多步骤复杂任务** — 需要拆解为 3 个以上子步骤才能完成
2. **规划—执行—审查—修正闭环任务** — 输出质量需要多轮迭代保证
3. **多模型互评或跨领域判断** — 单一模型不足以覆盖所有判断维度
4. **工具调用密集型** — 需要代码执行、联网检索、文件解析、数据库查询等
5. **可追溯性要求高** — 中间过程和决策理由需要完整记录

#### 不适用任务

1. **简单问答** — 单轮知识性问答不需要多 agent
2. **单轮改写/翻译** — 无需审查和迭代的任务
3. **低风险格式转换** — JSON 转 XML、Markdown 排版等
4. **不需要工具也不需要审查的轻量任务**
5. **用户明确要求快速直接回答的任务**

#### 协同触发规则

```
若任务复杂度 < 阈值                         → 不启用多 agent，单模型直接回答
若任务复杂度 >= 阈值，且存在多步骤/工具调用/校验需求/高风险输出  → 启用 DR-MMA
```

复杂度评估由下一节的 Task Complexity Evaluator 负责。

---

### 任务复杂度评估器 Task Complexity Evaluator

任务接入层中设置独立的复杂度评分模块，以评分驱动运行模式选择。

#### 评分模型

```
TaskComplexity =
  a1 * StepCount
  + a2 * DomainDepth
  + a3 * ToolRequirement
  + a4 * VerificationNeed
  + a5 * OutputRisk
  + a6 * ContextLength
```

#### 指标定义

| 指标 | 含义 | 评分范围 |
|------|------|---------|
| StepCount | 任务需要多少步骤才能完成 | 1~3 |
| DomainDepth | 是否涉及专业领域知识 | 0~3 |
| ToolRequirement | 是否需要代码、检索、文件、数据库等工具 | 0~3 |
| VerificationNeed | 是否需要事实、逻辑、数值校验 | 0~2 |
| OutputRisk | 输出错误是否会造成严重后果 | 0~2 |
| ContextLength | 是否涉及长上下文或多文件 | 0~2 |

#### 模式映射

| 分数范围 | 运行模式 | 说明 |
|---------|---------|------|
| 0~2 分 | Direct Mode | 单模型直接回答，不启用多 agent |
| 3~5 分 | Single Review Mode | 单模型生成 + 自审 |
| 6~8 分 | Compact Mode | 2~3 模型协同 |
| 9~12 分 | Standard Mode | 4~7 模型协同 |
| 12 分以上 | Expanded Mode | 专业 agent 并行协同 |

此评分机制与后文的五种运行模式直接衔接，使系统能**自动选择**合适的协同等级。

---

### 总体架构（九层）

```
Layer 1: 任务接入层 Task Intake Layer
  └─ 意图识别 / 约束提取 / 复杂度评估 / 风险判断 / 输出格式识别

Layer 2: 主控层 Supervisor Control Layer
  └─ 任务理解 / 任务图生成 / 角色需求判断 / 冲突裁决 / 最终审定

Layer 3: 任务图管理层 Task DAG Management Layer
  └─ 子任务节点 / 任务依赖 / 执行状态 / 超时处理 / 动态重规划

Layer 4: 动态角色管理层 Dynamic Role Management Layer
  └─ 模型池 / 能力注册表 / 角色模板库 / 角色合并/拆分 / 运行时绑定

Layer 5: 多 Agent 协同层 Agent Collaboration Layer
  └─ Planner / Executor / Researcher / Domain Expert / Critic / Verifier / Synthesizer

Layer 6: 通信与记忆层 Communication and Memory Layer
  └─ Blackboard / Event Bus / Debate Room / Decision Log / Artifact Store

Layer 7: 工具执行层 Tool Layer
  └─ 代码执行 / Web检索 / 文件解析 / 数据库查询 / 向量知识库 / 仿真接口

Layer 8: 安全与权限层 Permission and Safety Layer
  └─ 权限矩阵 / 操作分级 / 人工确认 / 审计日志

Layer 9: 可观测层 Observability Layer
  └─ DAG可视化 / 角色分配面板 / 事件流 / 成本追踪 / 质量评分
```

相比原始七层架构，新增了**安全与权限层**和**可观测层**，共九层。

---

### 八类标准 Agent 角色

| 角色 | 职责 | 关键能力需求 |
|------|------|-------------|
| **Supervisor** | 总控、裁决、调度、审定 | reasoning, synthesis, decision |
| **Planner** | 任务拆解、路线设计 | planning, reasoning |
| **Researcher** | 检索资料、整理证据 | research, tool_use |
| **Domain Expert** | 专业领域判断 | domain_knowledge, reasoning |
| **Executor** | 执行具体任务、编码 | coding, tool_use |
| **Critic** | 批判性审查、发现缺陷 | critic, reasoning |
| **Verifier** | 事实/逻辑/数值/格式校验 | verification, reasoning |
| **Synthesizer** | 最终整合、表达统一 | synthesis, writing |

---

### Task Contract 子任务合同

当前方案有任务 DAG、角色分配和黑板机制，但还需要一个统一的**任务合同**作为 agent 执行子任务的标准输入。每个任务节点生成一份 Task Contract：

```json
{
  "task_id": "T-003",
  "task_name": "设计动态角色分配机制",
  "task_type": "architecture_design",
  "objective": "提出模型数量不固定条件下的角色分配方法",
  "input_refs": ["BB-001", "BB-002"],
  "assigned_role": "Planner",
  "required_capabilities": ["planning", "reasoning"],
  "allowed_tools": ["blackboard_read", "blackboard_write"],
  "expected_output_schema": "role_assignment_design",
  "success_criteria": [
    "说明模型池、能力注册表、角色模板库之间的关系",
    "给出角色合并和拆分规则",
    "说明模型失败后的替换逻辑"
  ],
  "timeout_seconds": 120,
  "review_required": true,
  "contract_version": "1.0"
}
```

Task Contract 解决了 agent 拿到子任务后**输出边界不清晰、容易自由发挥**的问题。每个 agent 明确知道自己需要交付什么、用什么工具、达到什么标准。

---

### Agent 输出协议

Agent 不进行自由文本输出，而是按统一 Schema 将结果写入黑板：

```json
{
  "task_id": "T-003",
  "role_id": "R-Planner-01",
  "contract_version": "1.0",
  "status": "completed",
  "summary": "已完成动态角色分配机制设计",
  "claims": [
    {
      "claim": "角色不应固定绑定模型，而应运行时动态绑定",
      "evidence_refs": ["BB-001"],
      "confidence": 0.92
    }
  ],
  "artifacts": [
    {
      "artifact_type": "design_section",
      "artifact_id": "ART-014"
    }
  ],
  "risks": [
    {
      "risk": "能力评分不准确会导致角色分配错误",
      "severity": "medium",
      "mitigation": "引入能力校准任务和历史表现更新机制"
    }
  ],
  "next_action_recommendation": "交由 Critic 和 Verifier 审查"
}
```

此协议保证 Supervisor、Critic、Verifier 能够以统一格式稳定读取和比较不同 agent 的输出，是实现后续自动化审查和汇总的基础。

---

### 动态角色分配机制

#### 评分函数

决定将哪个模型分配到哪个角色：

```
Score(model, role) = w1 * CapabilityFit
                    + w2 * Reliability
                    + w3 * ToolFit
                    + w4 * ContextFit
                    - w5 * Cost
                    - w6 * Latency
```

**分配优先级**：先 Supervisor，再高风险角色（Critic/Verifier），再执行角色（Executor/Researcher），最后格式化角色。

#### 模型能力画像更新机制 Capability Calibration

能力分数由三部分组成，不依赖单一标注源：

```
CapabilityScore = 0.4 * BenchmarkScore
                 + 0.4 * HistoricalTaskScore
                 + 0.2 * HumanFeedbackScore
                 - FailurePenalty
```

各分量说明：

| 来源 | 说明 | 更新频率 |
|------|------|---------|
| BenchmarkScore | 标准测试任务表现 | 每次模型版本更新 |
| HistoricalTaskScore | 历史项目中的完成质量 | 每次任务完成后 |
| HumanFeedbackScore | 用户或管理员反馈 | 按反馈提交 |
| FailurePenalty | 频繁失败时降低对应能力分 | 每次失败事件 |

能力注册表扩展字段：

```sql
capability_registry (
    model_id,
    capability_type,
    score,
    confidence,          -- 置信度：样本量越大的 score 越可信
    sample_count,        -- 参与评分的历史任务数
    failure_count,       -- 失败次数（用于 FailurePenalty）
    last_evaluated_at,
    update_source        -- benchmark | history | feedback
)
```

`confidence` 字段很关键。一个模型在 coding 上得分 0.90 但只测试过 2 个任务，与测试过 200 个任务的 0.90 不能等价。

---

### 角色合并规则（模型不足时）

| 可合并组合 | 不建议合并 |
|-----------|-----------|
| Supervisor + Planner | Executor + Critic |
| Supervisor + Synthesizer | Researcher + Verifier |
| Critic + Verifier | Supervisor + Executor |
| Researcher + Summarizer | — |
| Domain Expert + Verifier | — |

核心原则：**生成与审查必须分离**，避免自我确认偏差。

### 角色拆分规则（模型充足时）

| 角色 | 可拆分为 |
|------|---------|
| Critic | Logic Critic + Risk Critic + Domain Critic |
| Researcher | Search Agent + Reading Agent + Evidence Agent |
| Coder | Code Generator + Code Reviewer + Test Agent |
| Domain Expert | Electromagnetic Expert + Control Expert + Dynamics Expert |

角色拆分必须服务任务目标，不应为增加 agent 数量而强行拆分。

---

### 通信机制三板斧

#### 1. 共享黑板 Blackboard

Agent 不进行长篇自由对话，而是将结构化结果写入黑板。内容包含：任务状态、中间结论、模型输出、证据材料、风险提示、冲突记录、工具结果。

#### 2. 事件总线 Event Bus

执行中 agent 发出事件信号，系统根据事件所需能力动态选择处理者：

| 事件类型 | 含义 | 触发处理 |
|---------|------|---------|
| TASK_COMPLETED | 子任务完成 | Supervisor 检查下一步就绪任务 |
| CONFLICT_DETECTED | 发现冲突 | Supervisor 判断类型，必要时发起 Debate Room |
| LOW_CONFIDENCE | 结论置信度不足 | Supervisor 决定是否复核或补充检索 |
| NEED_EVIDENCE | 需要补充证据 | 分配给 Researcher |
| TOOL_FAILED | 工具调用失败 | Supervisor 决定重试/降级/切换工具 |
| NEED_REPLAN | 需要重新规划 | 触发 Planner 重拆任务 DAG |
| NEED_REVIEW | 需要审查 | 通知 Critic 或 Verifier |
| HUMAN_CHECKPOINT | 需要人工确认 | 暂停任务流，等待用户确认 |

#### 3. 受控讨论室 Debate Room

针对高不确定性或冲突问题，Supervisor 发起有限轮次的限定角色讨论，最终由 Supervisor 裁决。

讨论规则：
- 由 Supervisor 发起
- 限定参与角色和讨论目标
- 限定轮数和输出格式
- 最终由 Supervisor 裁决并记录到 Decision Log

---

### 上下文管理机制

多 agent 系统最容易出问题的是上下文膨胀。明确**哪些内容进入上下文，哪些只存储不注入**。

#### 三类上下文分级

| 级别 | 内容 | 谁读取 | 管理策略 |
|------|------|--------|---------|
| **Runtime Context** | 当前子任务必须的最小上下文：任务目标、直接依赖结果、相关黑板条目、当前合同、输出格式 | 执行 agent | 每次调用时动态组装 |
| **Global Context** | 完整任务 DAG、所有子任务状态、事件日志、决策日志、关键风险记录 | Supervisor 专用 | Supervisor 独立维护，不注入执行 agent |
| **Artifact Context** | 大文件、中间代码、长文档、检索资料 | 所有角色 | 只保存引用（artifact_id, summary, checksum），不直接注入 |

#### 核心原则

```
Agent 默认只读取与当前 Task Contract 直接相关的上下文；
禁止将完整黑板无差别注入每个 agent。
```

否则系统越运行越慢，且上下文污染严重。

---

### 权限与安全控制 Permission and Safety Layer

未来接入文件、数据库、代码执行、联网检索后，需要明确不同 agent 的权限边界。

#### 权限矩阵

| 操作 | Planner | Executor | Researcher | Critic | Verifier | Supervisor |
|------|:-------:|:--------:|:----------:|:------:|:--------:|:----------:|
| 读取黑板 | Y | Y | Y | Y | Y | Y |
| 写入黑板 | Y | Y | Y | Y | Y | Y |
| 修改任务 DAG | N | N | N | N | N | Y |
| 调用代码沙箱 | N | Y | N | N | Optional | Y |
| 调用外部 API | N | Optional | Y | N | N | Y |
| 删除文件 | N | N | N | N | N | Human Only |
| 发送外部消息 | N | N | N | N | N | Human Only |

#### 三级操作分级

| 级别 | 定义 | 处理方式 |
|------|------|---------|
| **Safe Action** | 不产生外部影响，不消耗外部资源 | 自动执行 |
| **Risky Action** | 产生外部影响但可回滚 | 需 Supervisor 审批 |
| **Critical Action** | 产生不可逆外部影响 | 需人工确认节点 |

---

### 成本与预算控制器 Budget Controller

多模型协同系统中，Debate Room、Critic、Verifier、重试机制叠加后成本容易失控。

#### 预算定义

每个任务初始化时生成预算：

```json
{
  "task_id": "TASK-001",
  "budget": {
    "max_model_calls": 30,
    "max_tool_calls": 10,
    "max_debate_rounds": 2,
    "max_retries_per_node": 2,
    "max_total_tokens": 120000,
    "allow_high_cost_model": true,
    "budget_owner": "task-TASK-001"
  }
}
```

#### 追踪指标

```
BudgetTracker:
  model_calls_used: 12 / 30
  tool_calls_used: 4 / 10
  tokens_consumed: 45200 / 120000
  high_cost_calls: 2
  estimated_cost: $0.18
  status: "within_budget" | "warning" | "exceeded"
```

当追踪器进入 "warning" 状态时，Budget Controller 向 Supervisor 发送 `BUDGET_WARNING` 事件，由 Supervisor 决定是否申请追加预算或降低协同等级。

---

### Supervisor 防过载设计

Supervisor 权力较大——任务拆解、角色分配、流程调度、冲突裁决和最终审定都集中于此。同一个 prompt 承载全部职责会导致上下文压力过高。

#### 逻辑子模块拆分

Supervisor 拆分为独立子模块，每个模块有专用 prompt 和输入范围：

```
Supervisor =
  Task Understanding Module      — 解析任务目标，提取约束
  + DAG Planning Module          — 生成和调整任务 DAG
  + Role Assignment Controller   — 调用 Role Manager 分配角色
  + Event Handling Controller    — 处理事件总线信号
  + Decision Module              — 裁决冲突和争议
  + Final Review Module          — 最终审定输出质量
```

#### 分阶段加载策略

```
任务开始时 → 只加载 Task Understanding Module + DAG Planning Module
任务执行中 → 只加载 Event Handling Controller（持续监听）
事件触发时 → 按需加载 Decision Module 或 Role Assignment Controller
汇总阶段   → 加载 Final Review Module
```

在实现上各模块仍由同一个强模型承担，但 prompt 和上下文分阶段隔离，避免主控模型上下文压力持续累积。

---

### 完整执行流程

```
Step 1:  用户提交任务
Step 2:  任务接入层解析目标、约束和输出要求
Step 3:  Task Complexity Evaluator 评分，选择运行模式
Step 4:  Supervisor 判断是否启用多模型协同
Step 5:  Planner 生成初始任务拆解
Step 6:  Supervisor 生成任务 DAG + 各节点 Task Contract
Step 7:  Budget Controller 初始化预算
Step 8:  Role Manager 判断所需角色槽位，查询模型池和能力注册表
Step 9:  系统进行角色分配、合并或拆分，生成运行时 Agent
Step 10: 各 Runtime Agent 按 Task Contract 执行子任务
Step 11: Agent 将结构化结果按输出协议写入 Blackboard
Step 12: Event Bus 监听任务完成、冲突、低置信度和失败事件
Step 13: Supervisor 根据事件动态调整任务 DAG 或触发受控讨论
Step 14: Critic 和 Verifier 对关键结果进行审查
Step 15: Executor 根据审查意见修正结果（生成新版本，不覆盖）
Step 16: Synthesizer 按输出协议整合最终结果
Step 17: Supervisor 最终审定并返回用户
Step 18: 任务结果写入 Decision Log 和 Artifact Store
```

### 五种运行模式

| 模式 | 模型数 | 结构 | 复杂度分 | 适用场景 |
|------|--------|------|---------|---------|
| Direct | 1 | 单模型直接回答 | 0~2 | 简单问答、格式转换 |
| Single Review | 1 | 单模型多阶段自审 (Plan→Execute→Critique→Verify) | 3~5 | 低风险多步骤任务 |
| Compact | 2~3 | M1=Supervisor+Planner+Synthesizer, M2=Executor+Researcher, M3=Critic+Verifier | 6~8 | 最低有效多模型 |
| Standard | 4~7 | 各角色独立分配 | 9~12 | 推荐常规模式 |
| Expanded | 8+ | 增加专业 agent 和并行 agent | 12+ | 复杂专业任务 |

---

### 冲突处理

五类冲突：事实冲突、逻辑冲突、方案冲突、数值冲突、约束冲突。

处理流程：

```
Agent 发现冲突
  ↓
发出 CONFLICT_DETECTED 事件
  ↓
Supervisor 判断冲突类型和严重程度
  ↓
必要时发起 Debate Room（限定角色、轮数、格式）
  ↓
Verifier 核查事实或数值
  ↓
Critic 评估风险
  ↓
Supervisor 形成裁决
  ↓
Decision Log 记录裁决依据和分歧保留
```

处理原则：
- 可验证事实 > 模型观点
- 工具计算结果 > 自然语言推断
- 原始数据 > 二次总结
- 高置信度+有证据 > 低置信度
- Supervisor 拥有最终裁决权

若冲突无法完全解决，系统不应强行给出单一结论，而应保留分歧并说明：可确定的部分、存在分歧的部分、不同方案的依据、当前无法裁决的原因、推荐的保守选择。

---

### 失败恢复

| 机制 | 说明 |
|------|------|
| 超时机制 | 每个任务节点设最大运行时间，超时进入 TIMEOUT 状态 |
| 重试机制 | 有限次数重试（2~3 次），避免无限循环 |
| 备用模型 | 按能力注册表选择替补，Role Manager 重新分配 |
| 任务降级 | 资源不足时降低协同复杂度（Expanded → Standard → Compact） |
| 任务重拆 | 过大任务由 Supervisor 进一步拆解为更小子任务 |
| 人工确认 | 高风险操作设置 Human Checkpoint 节点 |

---

### Artifact 版本管理与回滚

系统产生大量中间结果：黑板条目、任务输出、代码、文档片段、决策日志等。每个 Artifact 应有版本控制。

#### Artifact 表结构

```sql
artifact_store (
    artifact_id,
    task_id,
    artifact_type,      -- design_doc | code_snippet | report | risk_analysis
    content_uri,        -- 存储位置（文件路径或对象存储 key）
    version,            -- 版本号，从 1 开始递增
    parent_version,     -- 基于哪个版本修改（null 表示初版）
    created_by_role,    -- 创建者角色
    status,             -- draft | under_review | verified | superseded
    checksum,           -- 内容校验和
    created_at
)
```

#### 版本演化规则

```
ART-001-v1：Executor 初稿
ART-001-v2：根据 Critic 意见修正（parent_version = v1）
ART-001-v3：Verifier 通过版本（status = verified）
ART-001-v4：Supervisor 审定终版（status = superseded, final = true）
```

每次修正生成新版本，不覆盖旧版本。这样可以追溯：
- 为什么最终采用这个结果？
- 哪一版被否定了？
- 谁提出了修改意见？
- Supervisor 为什么裁决？

---

### 可观测性与调试面板

多 agent 系统开发时，没有可视化调试界面很难定位问题。

#### 最小调试面板内容

```
1. 当前 Task DAG 可视化（节点状态着色）
2. 每个任务节点的状态、负责人、耗时
3. 当前运行时角色分配（模型 ↔ 角色映射）
4. 模型调用次数和累计成本
5. 黑板最新条目流（实时更新）
6. 事件总线实时流（事件类型 + 来源 + 处理状态）
7. 冲突处理记录（冲突类型 + 裁决 + 时间）
8. Supervisor 决策日志
9. 失败和重试记录
10. 最终输出质量评分
```

#### 后端数据结构支撑

```
observability_events (
    event_id,
    event_type,       -- dag_update | role_assign | model_call | blackboard_write | conflict | decision
    payload,          -- JSON 事件详情
    created_at
)
```

前端以时间线 + 拓扑图形式呈现，帮助开发者在调试时快速定位问题环节。

---

## 实施路径

### 总体原则

**按闭环完整度推进，而非按功能完整度。** 先跑通最小闭环，再扩展能力；先验证核心假设，再投入完整工程。

核心假设是：**多模型分工 + 审查闭环 + 主控汇总是否真的优于单模型直接生成。** 如果这个假设通不过测试，继续优化架构没有意义。

---

### 第一阶段：MVP 最小可行系统

#### 范围收缩

MVP 不实现完整 DR-MMA，而是实现一个**收缩版**，暂缓以下模块：

```
⏳ 完整动态角色系统（能力向量、角色拆分/合并算法、模型健康检查）
⏳ 完整事件总线（Redis/NATS）
⏳ 受控讨论室 Debate Room
⏳ 完整工具沙箱与外部 API 系统
⏳ 完整可观测性前端面板
⏳ 完整预算控制器
⏳ 向量记忆系统
⏳ 多领域专业 Agent
```

#### 七个核心模块

按实现顺序排列：

```
Module 1: ModelAdapter       — 统一模型调用接口
Module 2: TaskContract       — 子任务合同 Schema + 生成
Module 3: RoleRunner         — 五个基础角色 prompt + 执行器
Module 4: Blackboard         — 黑板持久化（JSON 文件起步）
Module 5: DecisionLog        — 决策日志记录
Module 6: ArtifactStore      — 线性版本管理
Module 7: WorkflowEngine     — 固定闭环编排
```

#### Module 1: ModelAdapter

统一不同模型调用方式。

```python
# 输入
model_adapter.chat(
    messages=[ChatMessage(...)],
    model_name="local-27b",
    temperature=0.7,
) -> ModelResponse(content, token_usage, latency, status)
```

MVP 只支持 2~3 个模型，不追求广泛的模型兼容性。

#### Module 2: TaskContract

每个子任务生成固定 Schema 合同：

```json
{
  "task_id": "T-001",
  "task_name": "子任务名称",
  "role": "Worker",
  "objective": "完成子任务的具体目标",
  "input_refs": ["BB-xxx"],
  "success_criteria": ["标准1", "标准2"],
  "timeout_seconds": 120,
  "review_required": true
}
```

#### Module 3: RoleRunner

五个基础角色，MVP 阶段使用同一个 Runner 类，通过不同 prompt template 区分：

| 角色 | Prompt 定位 | 核心指令 |
|------|------------|---------|
| **Planner** | 任务拆解 | "将以下任务拆解为 3~5 个子任务，每个子任务给出目标和验收标准" |
| **Worker** | 执行子任务 | "根据 Task Contract 完成子任务，输出结构化结果" |
| **Critic** | 批判审查 | "对以下输出进行批判性审查，列出缺陷、风险和改进建议" |
| **Verifier** | 事实/逻辑校验 | "验证以下输出的事实正确性、逻辑一致性和格式完整性" |
| **Supervisor** | 汇总裁决 | "综合所有子任务结果和审查意见，形成最终输出" |

每个角色的 prompt 边界明确包含四部分：

```
你能做什么：[角色职责描述]
你不能做什么：[边界约束]
你必须输出什么格式：[AgentResponse Schema]
你什么时候触发风险提示：[低置信度 / 发现冲突 / 需要补充信息]
```

#### Module 4: Blackboard

先做本地 JSONL 持久化，不上复杂消息系统。每条记录包含：

```json
{
  "entry_id": "BB-001",
  "task_id": "T-001",
  "source_role": "Worker",
  "content_type": "task_output",
  "summary": "已完成数据分析子任务",
  "payload": { ... },
  "created_at": "2026-07-09T13:00:00Z"
}
```

#### Module 5: DecisionLog

记录 Supervisor 每次裁决：

```json
{
  "decision_id": "DEC-001",
  "task_id": "T-001",
  "decision_type": "conflict_resolution",
  "ruling": "采用 Worker 修正版 v2",
  "rationale": "修正版修复了 Critic 指出的边界条件缺失问题",
  "evidence_refs": ["ART-001-v2", "CR-001"],
  "created_at": "2026-07-09T13:05:00Z"
}
```

#### Module 6: ArtifactStore

MVP 只需要线性版本，不依赖 parent_version 链：

```json
{
  "artifact_id": "ART-001",
  "task_id": "T-001",
  "version": 2,
  "content": "...",
  "created_by_role": "Worker",
  "status": "draft",
  "created_at": "2026-07-09T13:03:00Z"
}
```

#### Module 7: WorkflowEngine

MVP 用**固定顺序**取代复杂 DAG。流程固定为：

```
intake
→ plan
→ assign_roles（简化版）
→ execute
→ critique
→ revise
→ verify
→ synthesize
→ final_review
→ write_results
```

#### 简化版角色分配

MVP 不实现能力评分驱动的动态分配，采用固定规则：

```text
若模型数 = 1：
  同一模型顺序扮演 Planner / Worker / Critic / Verifier / Supervisor

若模型数 = 2：
  M1 = Supervisor + Planner + Synthesizer
  M2 = Worker + Critic + Verifier

若模型数 >= 3：
  M1 = Supervisor + Planner + Synthesizer
  M2 = Worker
  M3 = Critic + Verifier
```

先跑通闭环，再扩展为能力评分驱动的动态分配。

#### MVP 状态机

最小状态集，不多引入：

```
CREATED → PLANNED → ASSIGNED → RUNNING → REVIEWING → REVISING → VERIFYING → COMPLETED
                                                                          → FAILED
```

#### 固定三个 Schema

实现编码前必须先固定以下三个协议，避免后续频繁改动：

```python
# TaskContract Schema
{
  "task_id": str,
  "task_name": str,
  "role": str,                    # Planner | Worker | Critic | Verifier | Supervisor
  "objective": str,
  "input_refs": list[str],
  "success_criteria": list[str],
  "timeout_seconds": int,
  "review_required": bool
}

# AgentResponse Schema
{
  "task_id": str,
  "role": str,
  "status": str,                  # completed | failed | need_review | low_confidence
  "summary": str,
  "claims": list[{"claim": str, "confidence": float, "evidence_refs": list[str]}],
  "artifacts": list[{"artifact_id": str, "version": int}],
  "risks": list[{"risk": str, "severity": str, "mitigation": str}],
  "next_action_recommendation": str
}

# BlackboardEntry Schema
{
  "entry_id": str,
  "task_id": str,
  "source_role": str,
  "content_type": str,            # task_output | critic_report | verification_report | decision
  "summary": str,
  "payload": dict,
  "created_at": str
}
```

#### 实现顺序

严格按照以下 8 步推进，每一步完成后再进入下一步：

```text
Step 1: 定义三个固定 Schema（TaskContract / AgentResponse / BlackboardEntry）
Step 2: 实现 ModelAdapter（统一模型调用）
Step 3: 实现五个基础 RoleRunner prompt
Step 4: 实现 Blackboard 持久化（JSONL 文件）
Step 5: 实现 WorkflowEngine 固定闭环（plan → execute → critique → revise → verify → synthesize）
Step 6: 实现 ArtifactStore 线性版本管理
Step 7: 实现 DecisionLog 记录
Step 8: 完成 10 个中等复杂任务测试
```

#### 验收标准

```
1. 支持 2~3 个模型协同完成任务
2. 支持 Planner / Worker / Critic / Verifier / Supervisor 五类角色
3. 三个固定 Schema 在实现过程中不修改
4. 每个任务均生成：Task Contract → 执行输出 → 审查意见 → 修正版 → 验证通过 → 最终汇总
5. 失败时输出明确错误状态（FAILED + 原因），不静默失败
6. Artifact 版本号递增可查（v1 → v2 差异可追溯）
7. Decision Log 至少记录 Supervisor 的关键裁决
8. 所有中间过程可查询、可回放
```

#### 与现有 Symposium 代码的关系

MVP 基于 `symposium/` 代码库演化，但不直接继承其内部实现：

| Symposium 组件 | MVP 处理 | 原因 |
|---------------|---------|------|
| `core/message_bus.py` | 暂不直接使用，MVP 用结构化文件替代 | 事件总线是第三阶段内容 |
| `core/deliberation.py` | 暂不直接使用，MVP 用 Critic + Verifier 替代 | Debate Room 是第三阶段内容 |
| `core/synthesizer.py` | 提取 Supervisor 汇总逻辑思路 | 需重构为 Task Contract 驱动 |
| `core/executor.py` | 提取协同执行的基本思路 | 需重构为 RoleRunner 模式 |
| `core/workflow.py` | 提取流程编排骨架 | 需重构为固定 WorkflowEngine |
| `models/base.py` | 复用 ModelAdapter 抽象 | 已满足需求 |

主体代码**重新实现**，因为原有 Symposium 的重点在"多模型自由讨论"，而 DR-MMA MVP 的重点在"结构化合同驱动的审查闭环"。

---

### 第二阶段：动态角色管理

**目标**：解决模型数量不固定问题。

**实现模块**：
- 模型池 Model Pool — 管理可用模型的注册、状态和健康检查
- 能力注册表 Capability Registry — 能力向量维护（BenchmarkScore + HistoricalTaskScore + HumanFeedbackScore）
- 能力校准任务 — 定期运行标准测试更新能力分数
- 角色模板库 Role Template Library — 标准角色定义（含所需能力、可合并标记）
- 角色合并算法 — 模型不足时按合并规则表自动合并
- 角色拆分算法 — 模型充足时按拆分规则表自动拆分
- 运行时角色绑定 — 基于评分函数的模型→角色分配
- 故障转移 — 模型失败后的角色重分配

**验收标准**：
1. 支持 1 / 2 / 3 / 5 / 8 个模型数量下自动切换运行模式
2. 角色合并与拆分结果可解释、可记录
3. 模型失效后 30 秒内完成角色重分配
4. 能力评分有 confidence 字段，区分低样本和高样本评分的可信度
5. 角色分配记录可复现

---

### 第三阶段：通信与记忆基础设施

**目标**：注入运行期沟通能力，使系统具备实时协同和动态调整能力。

**实现模块**：
- 共享黑板持久化（Blackboard Storage）— 结构化条目 + 查询接口
- 事件总线（Event Bus）— 事件发布/订阅/路由
- 受控讨论室（Debate Room）— 限定角色/轮数/格式的受控讨论
- 决策日志持久化（Decision Log）— 可追溯的裁决记录
- Supervisor 防过载 — 逻辑子模块拆分和分阶段加载
- 上下文管理 — Runtime Context / Global Context / Artifact Context 三级分级策略

**验收标准**：
1. 支持 TASK_COMPLETED / CONFLICT_DETECTED / LOW_CONFIDENCE / TOOL_FAILED 等事件
2. 事件可触发任务重规划或受控讨论
3. 受控讨论室支持最多 N 轮讨论（N 可配置，默认 2）
4. 黑板条目支持按 task_id 和 content_type 检索
5. 决策日志支持按时间线回放
6. Supervisor 上下文压力有明确指标监控

---

### 第四阶段：工具与记忆接入

**目标**：使系统具备处理真实复杂任务的能力。

**接入能力**：
- 文件解析（PDF、图片、代码仓库）
- 代码执行沙箱（隔离且安全）
- 联网检索（Web Search + 文献数据库）
- 数据库查询（只读接口）
- 向量知识库（历史任务记忆）
- 中间产物存储与管理（Artifact Store）

**验收标准**：
1. 文件解析结果能进入 Artifact Store，不丢失元数据
2. 代码执行必须隔离在沙箱中，无法访问宿主系统
3. 工具调用失败能触发 TOOL_FAILED 事件并走失败恢复流程
4. 工具输出必须可被 Verifier 复核
5. 外部高风险操作必须进入 HUMAN_CHECKPOINT 节点

---

### 第五阶段：领域专业化

**目标**：进入专业化应用阶段。

**扩展方向**（按需选配）：
- 论文写作 Agent — 文献综述、摘要生成、格式校对
- 代码开发 Agent — 多语言代码生成、审查、测试
- 工程仿真 Agent — 电磁场、控制、动力学等专业仿真
- 数据分析 Agent — 统计建模、可视化、报告生成
- 知识管理 Agent — 文档摘要、知识图谱、问答系统

**验收标准**：
1. 每个专业 agent 有独立的能力画像和能力校准任务
2. 专业 agent 与通用 agent 可混合编排
3. 新增 agent 不影响已有协同流程

---

### 技术栈建议

| 组件 | 推荐方案 |
|------|---------|
| 后端框架 | Python + FastAPI |
| 任务编排 | LangGraph / 自研 DAG Engine |
| 消息机制 | Redis Streams / NATS |
| 状态存储 | PostgreSQL |
| 短期缓存 | Redis |
| 向量记忆 | Qdrant / pgvector |
| 文件存储 | MinIO / S3 |
| 日志观测 | OpenTelemetry |
| 可观测面板 | React / Vue + D3.js 拓扑图 |

### 数据库核心表结构

```sql
-- 模型注册表
model_registry (model_id, name, provider, context_length, cost_level, latency_level, status)

-- 模型能力表（含置信度）
capability_registry (model_id, capability_type, score, confidence, sample_count,
                     failure_count, last_evaluated_at, update_source)

-- 角色模板表
role_template (role_type, responsibility, required_capabilities_json,
               can_merge, can_split, merge_rules_json, split_rules_json)

-- 运行时角色分配（核心动态表）
runtime_role_assignment (task_id, role_id, role_type, assigned_model_id,
                         merged_roles_json, required_capabilities_json,
                         start_time, end_time, status)

-- 任务节点表
task_node (task_id, parent_task_id, dag_id, title, description,
           status, assigned_role, deadline, contract_json)

-- 事件日志表
event_log (event_id, task_id, event_type, source_role, priority,
           summary, payload_json, created_at, handled_by, resolution)

-- 共享黑板表
blackboard_entry (entry_id, task_id, source_role, content_type,
                  summary, claims_json, recommendation, created_at)

-- 决策日志表
decision_log (decision_id, task_id, conflict_type, ruling,
              rationale, evidence_refs_json, created_at, is_final)

-- 中间产物表
artifact_store (artifact_id, task_id, artifact_type, content_uri,
                version, parent_version, created_by_role, status,
                checksum, created_at)

-- 预算追踪表
budget_tracker (task_id, max_model_calls, model_calls_used,
                max_tool_calls, tool_calls_used,
                max_tokens, tokens_consumed,
                estimated_cost, budget_status, updated_at)
```

---

## 附录

### 与 Symposium 的关系

本项目的 MVP（第一阶段）已通过 `symposium/` 代码库实现验证：

- `core/message_bus.py` — 实现了事件总线的雏形
- `core/deliberation.py` — 实现了受控讨论的雏形
- `core/synthesizer.py` — 实现了 Supervisor 汇总决策的雏形
- `core/executor.py` — 实现了协同执行和实时沟通的雏形
- `core/workflow.py` — 实现了全流程编排的雏形

后续各阶段将在现有 Symposium 框架基础上逐步补齐 DR-MMA 完整能力：动态角色管理、能力注册表、角色模板库、共享黑板持久化、版本管理、预算控制器、安全权限层和可观测面板。

### 方案特点总结

| 维度 | 特点 |
|------|------|
| 适应性 | 模型数量 1~N 均可运行，通过角色合并/拆分自动适配 |
| 可控性 | 所有交流通过黑板、事件总线和受控讨论室完成，避免自由群聊混乱 |
| 可追溯 | 中间结果、冲突记录、裁决过程完整记录，支持复盘和回滚 |
| 可靠性 | Critic + Verifier + Supervisor 三级审查，降低错误输出风险 |
| 工程可实现 | 从 5 角色 MVP 起步，逐阶段扩展，每个阶段有明确验收标准 |
| 成本控制 | Budget Controller 监控全链路调用消耗，防止成本失控 |
| 安全约束 | 三级操作分级 + 权限矩阵，高风险操作必须人工确认 |
| 上下文防护 | Runtime / Global / Artifact 三级上下文隔离，防止膨胀 |
