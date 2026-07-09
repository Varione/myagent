# DR-MMA Desktop UI — 实施计划

## 目标
为 DR-MMA agent 框架开发一个 CustomTkinter 桌面端 UI，实现任务输入、工作流可视化、结果展示、日志查看和模型配置功能。

## 架构设计

### 分层
```
UI Layer (CustomTkinter widgets)
    ↓ 事件/回调
Controller Layer (threaded bridge)
    ↓ 调用
Engine Layer (WorkflowEngine, existing)
```

### 组件树
```
App
├── Sidebar (导航)
│   ├── Task (任务输入)
│   ├── Pipeline (流程可视化)
│   ├── Results (结果展示)
│   ├── Logs (日志)
│   └── Config (配置)
└── Content Area (CTkTabView)
```

### 视图说明
| 视图 | 功能 | 关键组件 |
|------|------|---------|
| Task | 任务输入框 + Execute 按钮 + 模型选择 | CTkTextbox, CTkButton, CTkOptionMenu |
| Pipeline | 工作流各阶段状态展示 | 自定义 CTkFrame 流水线卡片 |
| Results | 子任务输出 tab 页 | CTkTabview, CTkTextbox |
| Logs | Blackboard + DecisionLog 查询 | CTkTextbox, CTkSegmentedButton |
| Config | 模型端点/API Key/路径配置 | CTkEntry, CTkButton(save) |

## 实施步骤

### Phase 1: 基础框架
- [x] 创建 `dr_mma/ui/` 目录结构
- [x] 实现 `main_window.py` — 带侧边栏的主窗口
- [x] 实现 `app.py` — 入口点

### Phase 2: 核心面板
- [x] `task_panel.py` — 任务输入 + 执行
- [x] `pipeline_panel.py` — 流水线可视化
- [x] `results_panel.py` — 结果展示

### Phase 3: 辅助面板 + 集成
- [x] `log_panel.py` — 日志查看
- [x] `config_panel.py` — 模型配置
- [x] `controller.py` — 线程桥接 + 回调

### Phase 4: 调试 & 收尾
- [x] 验证可运行
- [x] 修复问题
- [x] 更新 `__init__.py`
