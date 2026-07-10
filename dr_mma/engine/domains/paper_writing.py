"""
Paper Writing Domain Agent — 论文写作专业 Agent。

遵循 IEEE TII 期刊论文写作规范：
- 六段式摘要结构
- 被动语态为主
- 量化结果优先
- 文献综述生成、摘要生成、格式校对三种子任务类型
"""

from __future__ import annotations

import re
from typing import Any, Optional

from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainTask,
    DomainType,
)


# ─── IEEE TII 写作规范常量 ───────────────────────────────────────────────

# 六段式摘要各段的关键词标记
ABSTRACT_SECTIONS = [
    "background",       # 宏观背景锚定
    "motivation",       # 研究动机/问题陈述
    "contribution",     # 本文工作总述
    "methodology",      # 分步描述研究方法
    "results",          # 核心结果呈现
    "significance",     # 工程/理论意义升华
]

# 论文写作技能列表
PAPER_WRITING_SKILLS = {
    "literature_review": 0.0,
    "abstract_writing": 0.0,
    "format_proofing": 0.0,
    "citation_check": 0.0,
    "structure_design": 0.0,
}

# 子任务类型常量
TASK_LITERATURE_REVIEW = "literature_review"
TASK_ABSTRACT_WRITING = "abstract_writing"
TASK_FORMAT_PROOFING = "format_proofing"


class PaperWritingAgent(DomainAgent):
    """
    论文写作领域专业 Agent。

    支持以下子任务类型：
    - literature_review: 文献综述生成
    - abstract_writing: 摘要生成（六段式）
    - format_proofing: 格式校对（IEEE TII 规范）
    """

    def __init__(self, agent_id: str = "paper_writer_1"):
        super().__init__(agent_id, DomainType.PAPER_WRITING)
        # 初始化技能画像
        self.profile.skills = dict(PAPER_WRITING_SKILLS)

    # ─── Abstract Methods ──────────────────────────────────────────────

    def get_domain_skills(self) -> dict[str, float]:
        """返回论文写作领域的标准技能列表及当前评分。"""
        return dict(self.profile.skills)

    def get_calibration_tasks(self) -> list[dict]:
        """返回校准任务定义（至少3个）。"""
        return [
            {
                "name": "calib_literature_review",
                "objective": "验证文献综述生成能力",
                "input": {
                    "topic": "evacuated tube transportation system",
                    "keywords": ["vacuum", "maglev", "hypersonic"],
                    "max_references": 10,
                },
            },
            {
                "name": "calib_abstract_writing",
                "objective": "验证六段式摘要生成能力",
                "input": {
                    "title": "A Novel Permanent Magnet Eddy Current Brake for Evacuated Tube Transportation",
                    "key_findings": [
                        "braking distance decreases by 15 percent",
                        "novel guideway configuration employed",
                    ],
                    "word_limit": 250,
                },
            },
            {
                "name": "calib_format_proofing",
                "objective": "验证 IEEE TII 格式校对能力",
                "input": {
                    "text": "Recently, with the development of technology, we can see that AI is very important. The results are (very good). Next, we will show then our method.",
                    "style": "ieee_tii",
                },
            },
        ]

    def execute_domain_task(self, task: DomainTask) -> dict:
        """执行论文写作相关任务。"""
        task_type = self._resolve_task_type(task)

        if task_type == TASK_LITERATURE_REVIEW:
            return self._execute_literature_review(task)
        elif task_type == TASK_ABSTRACT_WRITING:
            return self._execute_abstract_writing(task)
        elif task_type == TASK_FORMAT_PROOFING:
            return self._execute_format_proofing(task)
        else:
            return {
                "status": "error",
                "task_id": task.task_id,
                "message": f"Unsupported task type: {task_type}",
            }

    def validate_output(self, output: dict) -> tuple[bool, list[str]]:
        """验证输出是否符合 IEEE TII 格式规范。"""
        issues = []

        if not isinstance(output, dict):
            return False, ["Output must be a dictionary"]

        status = output.get("status", "")
        if not status:
            issues.append("Missing 'status' field in output")

        task_id = output.get("task_id", "")
        if not task_id:
            issues.append("Missing 'task_id' field in output")

        # 根据任务类型进行特定验证
        task_type = output.get("task_type", "")

        if task_type == TASK_LITERATURE_REVIEW:
            issues.extend(self._validate_literature_review(output))
        elif task_type == TASK_ABSTRACT_WRITING:
            issues.extend(self._validate_abstract_writing(output))
        elif task_type == TASK_FORMAT_PROOFING:
            issues.extend(self._validate_format_proofing(output))

        passed = len(issues) == 0
        return passed, issues

    # ─── Task Type Resolution ──────────────────────────────────────────

    def _resolve_task_type(self, task: DomainTask) -> str:
        """从任务中解析子任务类型。"""
        # 优先从 metadata 中读取
        if "task_type" in task.metadata:
            return task.metadata["task_type"]

        # 从 task_name 推断
        name_lower = task.task_name.lower()
        if "literature" in name_lower or "review" in name_lower:
            return TASK_LITERATURE_REVIEW
        if "abstract" in name_lower:
            return TASK_ABSTRACT_WRITING
        if "format" in name_lower or "proof" in name_lower:
            return TASK_FORMAT_PROOFING

        # 从 objective 推断
        obj_lower = task.objective.lower()
        if "literature" in obj_lower or "综述" in task.objective:
            return TASK_LITERATURE_REVIEW
        if "abstract" in obj_lower or "摘要" in task.objective:
            return TASK_ABSTRACT_WRITING
        if "format" in obj_lower or "校对" in task.objective:
            return TASK_FORMAT_PROOFING

        # 默认返回文献综述
        return TASK_LITERATURE_REVIEW

    # ─── Literature Review ─────────────────────────────────────────────

    def _execute_literature_review(self, task: DomainTask) -> dict:
        """生成文献综述。"""
        topic = task.input_data.get("topic", "general")
        keywords = task.input_data.get("keywords", [])
        max_refs = task.input_data.get("max_references", 10)

        # 构建结构化综述内容
        sections = self._build_literature_sections(topic, keywords, max_refs)

        return {
            "status": "completed",
            "task_id": task.task_id,
            "task_type": TASK_LITERATURE_REVIEW,
            "topic": topic,
            "sections": sections,
            "reference_count": len(sections.get("references", [])),
            "structure_valid": True,
        }

    def _build_literature_sections(
        self, topic: str, keywords: list[str], max_refs: int
    ) -> dict:
        """构建文献综述的结构化内容。"""
        references = []
        for i in range(min(max_refs, 10)):
            ref = {
                "id": f"ref_{i + 1:03d}",
                "title": f"Study on {topic} using {keywords[i % len(keywords)] if keywords else 'advanced methods'}",
                "year": 2020 + (i % 6),
                "key_contribution": f"Proposed a novel approach for {topic}",
            }
            references.append(ref)

        return {
            "introduction": f"This section reviews the state of the art in {topic}.",
            "thematic_groups": self._group_references_by_theme(references),
            "research_gaps": [
                f"Limited investigation on high dimensional parameter spaces in {topic}",
                "Insufficient experimental validation under extreme operating conditions",
            ],
            "references": references,
        }

    def _group_references_by_theme(self, references: list[dict]) -> list[dict]:
        """将参考文献按主题分组。"""
        themes = [
            {
                "theme": "Theoretical Foundations",
                "refs": [r["id"] for r in references[:3]],
            },
            {
                "theme": "Experimental Studies",
                "refs": [r["id"] for r in references[3:6]],
            },
            {
                "theme": "Optimization Methods",
                "refs": [r["id"] for r in references[6:]],
            },
        ]
        return themes

    def _validate_literature_review(self, output: dict) -> list[str]:
        """验证文献综述输出。"""
        issues = []
        sections = output.get("sections", {})

        if not isinstance(sections, dict):
            issues.append("Sections must be a dictionary")
            return issues

        required_keys = ["introduction", "thematic_groups", "research_gaps", "references"]
        for key in required_keys:
            if key not in sections:
                issues.append(f"Missing required section: {key}")

        # 验证参考文献列表非空
        refs = sections.get("references", [])
        if not refs:
            issues.append("References list is empty")

        # 验证每个参考文献有基本字段
        for ref in refs:
            if "id" not in ref:
                issues.append("Reference missing 'id' field")
            if "title" not in ref:
                issues.append("Reference missing 'title' field")

        return issues

    # ─── Abstract Writing ──────────────────────────────────────────────

    def _execute_abstract_writing(self, task: DomainTask) -> dict:
        """生成六段式摘要。"""
        title = task.input_data.get("title", "Untitled")
        key_findings = task.input_data.get("key_findings", [])
        word_limit = task.input_data.get("word_limit", 250)

        abstract_parts = self._build_abstract_parts(title, key_findings, word_limit)
        full_abstract = self._assemble_abstract(abstract_parts)

        return {
            "status": "completed",
            "task_id": task.task_id,
            "task_type": TASK_ABSTRACT_WRITING,
            "title": title,
            "abstract": full_abstract,
            "sections": abstract_parts,
            "word_count": self._count_words(full_abstract),
            "within_limit": self._count_words(full_abstract) <= word_limit,
        }

    def _build_abstract_parts(
        self, title: str, key_findings: list[str], word_limit: int
    ) -> dict:
        """按六段式结构构建摘要各部分。"""
        # 1. 宏观背景锚定
        background = (
            f"Construction is underway for advanced transportation systems "
            f"that require innovative braking and control mechanisms."
        )

        # 2. 研究动机/问题陈述
        motivation = (
            "Nevertheless, conventional approaches tend to exhibit insufficient "
            "damping characteristics under high speed operating conditions."
        )

        # 3. 本文工作总述
        contribution = f"In this work, {title.lower()} is presented."

        # 4. 分步描述研究方法
        methodology_parts = []
        methodology_parts.append(
            "First, the structural configuration and operating principle are described."
        )
        methodology_parts.append(
            "Second, a mathematical model is established for theoretical analysis."
        )
        methodology_parts.append(
            "Finally, experimental validation is performed on a dedicated test bench."
        )
        methodology = " ".join(methodology_parts)

        # 5. 核心结果呈现（使用量化指标）
        if key_findings:
            results = "The results reveal that " + "; ".join(key_findings[:2]) + "."
        else:
            results = (
                "The results demonstrate a consistent performance improvement "
                "across the tested parameter range."
            )

        # 6. 工程/理论意义升华
        significance = (
            "This work presents an alternative scheme for practical deployment "
            "and provides theoretical insights into the underlying mechanisms."
        )

        return {
            "background": background,
            "motivation": motivation,
            "contribution": contribution,
            "methodology": methodology,
            "results": results,
            "significance": significance,
        }

    def _assemble_abstract(self, parts: dict) -> str:
        """将六段式摘要组装为完整文本。"""
        sentences = []
        for section in ABSTRACT_SECTIONS:
            text = parts.get(section, "")
            if text:
                sentences.append(text)
        return " ".join(sentences)

    def _count_words(self, text: str) -> int:
        """计算英文单词数。"""
        if not text:
            return 0
        return len(text.split())

    def _validate_abstract_writing(self, output: dict) -> list[str]:
        """验证摘要输出符合 IEEE TII 六段式规范。"""
        issues = []

        abstract = output.get("abstract", "")
        if not abstract:
            issues.append("Abstract text is empty")
            return issues

        sections = output.get("sections", {})
        if not isinstance(sections, dict):
            issues.append("Sections must be a dictionary")
            return issues

        # 检查六段式结构完整性
        missing_sections = []
        for section in ABSTRACT_SECTIONS:
            if section not in sections or not sections[section]:
                missing_sections.append(section)
        if missing_sections:
            issues.append(
                f"Missing abstract sections: {', '.join(missing_sections)}"
            )

        # 检查被动语态比例（IEEE TII 规范要求被动语态为主）
        passive_ratio = self._check_passive_voice_ratio(abstract)
        if passive_ratio < 0.3:
            issues.append(
                f"Passive voice ratio ({passive_ratio:.2f}) is below recommended threshold (0.3)"
            )

        # 检查是否包含量化指标
        has_quantification = bool(re.search(r"\d+\s*%", abstract))
        if not has_quantification:
            issues.append(
                "Abstract lacks quantitative results (IEEE TII prefers quantified metrics)"
            )

        # 检查字数限制
        word_count = output.get("word_count", self._count_words(abstract))
        word_limit = output.get("_word_limit_check", 250)
        if word_count > word_limit:
            issues.append(
                f"Word count ({word_count}) exceeds limit ({word_limit})"
            )

        return issues

    def _check_passive_voice_ratio(self, text: str) -> float:
        """
        估算被动语态比例。

        检测模式：be动词 + 过去分词（is presented, are performed, was conducted 等）。
        这是简化启发式方法，非精确 NLP 分析。
        """
        if not text:
            return 0.0

        # 常见被动语态模式
        passive_patterns = [
            r"\bis\s+\w+ed\b",           # is + V-ed
            r"\bare\s+\w+ed\b",           # are + V-ed
            r"\bwas\s+\w+ed\b",           # was + V-ed
            r"\bwere\s+\w+ed\b",          # were + V-ed
            r"\bhas\s+been\s+\w+ed\b",    # has been + V-ed
            r"\bhave\s+been\s+\w+ed\b",   # have been + V-ed
            r"\bhad\s+been\s+\w+ed\b",    # had been + V-ed
            r"\bis\s+presented\b",        # is presented
            r"\bis\s+employed\b",         # is employed
            r"\bis\s+utilized\b",         # is utilized
            r"\bare\s+performed\b",       # are performed
            r"\bare\s+described\b",       # are described
            r"\bcan\s+serve\b",           # can serve (semi-passive)
        ]

        total_sentences = len(re.split(r"[.!?]+", text))
        if total_sentences == 0:
            return 0.0

        passive_count = 0
        for pattern in passive_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            passive_count += len(matches)

        return min(1.0, passive_count / max(1, total_sentences))

    # ─── Format Proofing ───────────────────────────────────────────────

    def _execute_format_proofing(self, task: DomainTask) -> dict:
        """执行 IEEE TII 格式校对。"""
        text = task.input_data.get("text", "")
        style = task.input_data.get("style", "ieee_tii")

        corrections = self._proofread_ieee_tii(text)

        return {
            "status": "completed",
            "task_id": task.task_id,
            "task_type": TASK_FORMAT_PROOFING,
            "original_text": text,
            "corrected_text": corrections.get("corrected", text),
            "issues_found": corrections.get("issues", []),
            "issue_count": len(corrections.get("issues", [])),
            "style": style,
        }

    def _proofread_ieee_tii(self, text: str) -> dict:
        """
        按照 IEEE TII 写作规范进行格式校对。

        检查项：
        1. 空泛背景引入（如 "Recently, with the development of..."）
        2. 口语化表达（如 "we can see", "it is obvious"）
        3. 括号滥用
        4. 非正式连接词（如 "then", "next" 用作段落过渡）
        5. 连字符复合词（high-dimensional -> high dimensional）
        6. 主观副词过度使用
        """
        issues = []
        corrected = text

        # 检查1: 空泛背景引入
        vague_intro_patterns = [
            (r"(?i)recently,\s*with\s+the\s+development", "Vague background introduction detected"),
            (r"(?i)in\s+recent\s+years", "Consider anchoring with specific project data instead of 'in recent years'"),
        ]
        for pattern, message in vague_intro_patterns:
            if re.search(pattern, corrected):
                issues.append({"type": "vague_introduction", "message": message})

        # 检查2: 口语化表达
        colloquial_patterns = [
            (r"(?i)we\s+can\s+see", "Colloquial expression 'we can see' detected"),
            (r"(?i)it\s+is\s+obvious", "Avoid subjective expression 'it is obvious'"),
            (r"(?i)it\s+is\s+worth\s+noting", "Consider more formal alternative"),
        ]
        for pattern, message in colloquial_patterns:
            if re.search(pattern, corrected):
                issues.append({"type": "colloquial_expression", "message": message})

        # 检查3: 括号滥用（IEEE TII 要求去括号化）
        paren_count = corrected.count("(") - corrected.count("ref.") // 2
        if corrected.count("(") > 3:
            issues.append({
                "type": "excessive_parentheses",
                "message": f"Found {corrected.count('(')} parentheses; IEEE TII prefers converting parenthetical content to full sentences",
            })

        # 检查4: 非正式连接词
        informal_connectors = [
            (r"(?i)\bthen\s+,?\s*(we|the)", "Non-formal connector 'then' detected"),
            (r"(?i)\bnext\s+,?\s*(we|this)", "Non-formal connector 'next' detected; use 'Additionally' or 'Furthermore'"),
        ]
        for pattern, message in informal_connectors:
            if re.search(pattern, corrected):
                issues.append({"type": "informal_connector", "message": message})

        # 检查5: 连字符复合词（保留算法名称中的标准连字符）
        hyphenated_composites = [
            (r"\bhigh-dimensional\b", "high dimensional"),
            (r"\bmulti-objective\b", "multi objective"),
            (r"\btrade-off\b", "trade off"),
            (r"\breal-time\b", "real time"),
        ]
        for pattern, replacement in hyphenated_composites:
            if re.search(pattern, corrected):
                issues.append({
                    "type": "hyphen_composite",
                    "message": "Remove hyphen in " + pattern.replace(r"\b", "").replace("\\", "") + "; use " + replacement + " instead (unless part of algorithm name)",
                })

        # 生成修正后的文本
        for pattern, replacement in hyphenated_composites:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        return {
            "corrected": corrected,
            "issues": issues,
        }

    def _validate_format_proofing(self, output: dict) -> list[str]:
        """验证格式校对输出。"""
        issues = []

        if "original_text" not in output:
            issues.append("Missing 'original_text' field")

        if "corrected_text" not in output:
            issues.append("Missing 'corrected_text' field")

        if "issues_found" not in output:
            issues.append("Missing 'issues_found' field")

        # 验证 issues_found 是列表
        found = output.get("issues_found", None)
        if found is not None and not isinstance(found, list):
            issues.append("'issues_found' must be a list")

        return issues
