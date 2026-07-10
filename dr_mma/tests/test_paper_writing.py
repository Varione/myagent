"""Paper Writing Agent unit tests."""

import pytest
from dr_mma.engine.domain_agents import (
    DomainAgent,
    DomainRegistry,
    DomainType,
    CalibrationStatus,
    DomainTask,
)
from dr_mma.engine.domains.paper_writing import (
    PaperWritingAgent,
    PAPER_WRITING_SKILLS,
    TASK_LITERATURE_REVIEW,
    TASK_ABSTRACT_WRITING,
    TASK_FORMAT_PROOFING,
)


class TestPaperWritingAgentInitialization:
    def test_inherits_domain_agent(self):
        agent = PaperWritingAgent()
        assert isinstance(agent, DomainAgent)

    def test_domain_type_is_paper_writing(self):
        agent = PaperWritingAgent()
        assert agent.domain == DomainType.PAPER_WRITING

    def test_default_agent_id(self):
        agent = PaperWritingAgent()
        assert agent.agent_id == "paper_writer_1"

    def test_custom_agent_id(self):
        agent = PaperWritingAgent("my_writer")
        assert agent.agent_id == "my_writer"

    def test_profile_domain_matches(self):
        agent = PaperWritingAgent()
        assert agent.profile.domain == DomainType.PAPER_WRITING


class TestGetDomainSkills:
    def test_returns_all_five_skills(self):
        agent = PaperWritingAgent()
        skills = agent.get_domain_skills()
        assert len(skills) == 5

    def test_required_skills_present(self):
        agent = PaperWritingAgent()
        skills = agent.get_domain_skills()
        expected = {
            "literature_review",
            "abstract_writing",
            "format_proofing",
            "citation_check",
            "structure_design",
        }
        assert set(skills.keys()) == expected

    def test_initial_scores_are_zero(self):
        agent = PaperWritingAgent()
        skills = agent.get_domain_skills()
        for score in skills.values():
            assert score == 0.0

    def test_returns_dict(self):
        agent = PaperWritingAgent()
        skills = agent.get_domain_skills()
        assert isinstance(skills, dict)


class TestGetCalibrationTasks:
    def test_returns_at_least_three_tasks(self):
        agent = PaperWritingAgent()
        tasks = agent.get_calibration_tasks()
        assert len(tasks) >= 3

    def test_task_has_required_fields(self):
        agent = PaperWritingAgent()
        tasks = agent.get_calibration_tasks()
        for task in tasks:
            assert "name" in task
            assert "objective" in task
            assert "input" in task

    def test_covers_all_three_subtask_types(self):
        agent = PaperWritingAgent()
        tasks = agent.get_calibration_tasks()
        names = [t["name"] for t in tasks]
        # Should cover literature_review, abstract_writing, format_proofing
        assert any("literature" in n for n in names)
        assert any("abstract" in n for n in names)
        assert any("format" in n or "proof" in n for n in names)


class TestExecuteLiteratureReview:
    def _create_review_task(self, **kwargs):
        return DomainTask(
            task_id="T-review-001",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Literature Review",
            objective="Generate literature review",
            input_data={
                "topic": kwargs.get("topic", "evacuated tube transportation"),
                "keywords": kwargs.get("keywords", ["vacuum", "maglev"]),
                "max_references": kwargs.get("max_references", 5),
            },
        )

    def test_returns_completed_status(self):
        agent = PaperWritingAgent()
        task = self._create_review_task()
        result = agent.execute_domain_task(task)
        assert result["status"] == "completed"

    def test_task_type_is_literature_review(self):
        agent = PaperWritingAgent()
        task = self._create_review_task()
        result = agent.execute_domain_task(task)
        assert result["task_type"] == TASK_LITERATURE_REVIEW

    def test_contains_sections(self):
        agent = PaperWritingAgent()
        task = self._create_review_task()
        result = agent.execute_domain_task(task)
        assert "sections" in result
        sections = result["sections"]
        assert "introduction" in sections
        assert "thematic_groups" in sections
        assert "research_gaps" in sections
        assert "references" in sections

    def test_references_are_populated(self):
        agent = PaperWritingAgent()
        task = self._create_review_task(max_references=5)
        result = agent.execute_domain_task(task)
        refs = result["sections"]["references"]
        assert len(refs) == 5

    def test_reference_count_matches(self):
        agent = PaperWritingAgent()
        task = self._create_review_task(max_references=3)
        result = agent.execute_domain_task(task)
        assert result["reference_count"] == 3

    def test_respects_max_references(self):
        agent = PaperWritingAgent()
        task = self._create_review_task(max_references=20)
        result = agent.execute_domain_task(task)
        # Cap at 10 internal limit
        assert result["reference_count"] <= 10

    def test_thematic_groups_exist(self):
        agent = PaperWritingAgent()
        task = self._create_review_task()
        result = agent.execute_domain_task(task)
        groups = result["sections"]["thematic_groups"]
        assert len(groups) > 0
        for group in groups:
            assert "theme" in group
            assert "refs" in group

    def test_research_gaps_non_empty(self):
        agent = PaperWritingAgent()
        task = self._create_review_task()
        result = agent.execute_domain_task(task)
        gaps = result["sections"]["research_gaps"]
        assert len(gaps) > 0


class TestExecuteAbstractWriting:
    def _create_abstract_task(self, **kwargs):
        return DomainTask(
            task_id="T-abstract-001",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Abstract Writing",
            objective="Generate abstract for paper",
            input_data={
                "title": kwargs.get(
                    "title",
                    "A Novel Permanent Magnet Eddy Current Brake",
                ),
                "key_findings": kwargs.get(
                    "key_findings",
                    [
                        "braking distance decreases by 15 percent",
                        "novel guideway configuration employed",
                    ],
                ),
                "word_limit": kwargs.get("word_limit", 250),
            },
        )

    def test_returns_completed_status(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task()
        result = agent.execute_domain_task(task)
        assert result["status"] == "completed"

    def test_task_type_is_abstract_writing(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task()
        result = agent.execute_domain_task(task)
        assert result["task_type"] == TASK_ABSTRACT_WRITING

    def test_abstract_text_generated(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task()
        result = agent.execute_domain_task(task)
        assert len(result["abstract"]) > 0

    def test_six_section_structure(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task()
        result = agent.execute_domain_task(task)
        sections = result["sections"]
        expected_sections = {
            "background",
            "motivation",
            "contribution",
            "methodology",
            "results",
            "significance",
        }
        assert set(sections.keys()) == expected_sections

    def test_abstract_contains_methodology_markers(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task()
        result = agent.execute_domain_task(task)
        abstract = result["abstract"]
        # IEEE TII uses "First", "Second", "Finally" for methodology steps
        assert "First" in abstract
        assert "Second" in abstract

    def test_abstract_contains_key_findings(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task(
            key_findings=["braking distance decreases by 15 percent"]
        )
        result = agent.execute_domain_task(task)
        assert "15" in result["abstract"]

    def test_word_count_provided(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task()
        result = agent.execute_domain_task(task)
        assert "word_count" in result
        assert isinstance(result["word_count"], int)
        assert result["word_count"] > 0

    def test_within_limit_flag(self):
        agent = PaperWritingAgent()
        task = self._create_abstract_task(word_limit=500)
        result = agent.execute_domain_task(task)
        assert result["within_limit"] is True


class TestExecuteFormatProofing:
    def _create_proof_task(self, text, **kwargs):
        return DomainTask(
            task_id="T-proof-001",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Format Proofing",
            objective="Proofread paper format",
            input_data={
                "text": text,
                "style": kwargs.get("style", "ieee_tii"),
            },
        )

    def test_returns_completed_status(self):
        agent = PaperWritingAgent()
        task = self._create_proof_task("Some text to proofread.")
        result = agent.execute_domain_task(task)
        assert result["status"] == "completed"

    def test_task_type_is_format_proofing(self):
        agent = PaperWritingAgent()
        task = self._create_proof_task("Some text")
        result = agent.execute_domain_task(task)
        assert result["task_type"] == TASK_FORMAT_PROOFING

    def test_detects_vague_introduction(self):
        agent = PaperWritingAgent()
        task = self._create_proof_task(
            "Recently, with the development of technology, AI is popular."
        )
        result = agent.execute_domain_task(task)
        assert result["issue_count"] > 0

    def test_detects_colloquial_expression(self):
        agent = PaperWritingAgent()
        task = self._create_proof_text(
            "We can see that the results are very good."
        )
        result = agent.execute_domain_task(task)
        assert result["issue_count"] > 0

    def test_detects_hyphen_composite(self):
        agent = PaperWritingAgent()
        task = self._create_proof_task(
            "This is a high-dimensional optimization problem."
        )
        result = agent.execute_domain_task(task)
        assert result["issue_count"] > 0

    def test_corrected_text_provided(self):
        agent = PaperWritingAgent()
        task = self._create_proof_task("high-dimensional analysis")
        result = agent.execute_domain_task(task)
        assert "corrected_text" in result
        assert isinstance(result["corrected_text"], str)

    def test_clean_text_has_no_issues(self):
        agent = PaperWritingAgent()
        task = self._create_proof_task(
            "The braking distance consistently decreases by 15 percent."
        )
        result = agent.execute_domain_task(task)
        # This clean text should have no issues
        assert result["issue_count"] == 0

    def _create_proof_text(self, text):
        """Helper alias."""
        return DomainTask(
            task_id="T-proof-001",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Format Proofing",
            objective="Proofread paper format",
            input_data={"text": text, "style": "ieee_tii"},
        )


class TestValidateOutput:
    def test_validates_literature_review_success(self):
        agent = PaperWritingAgent()
        output = {
            "status": "completed",
            "task_id": "T-1",
            "task_type": TASK_LITERATURE_REVIEW,
            "sections": {
                "introduction": "Intro text",
                "thematic_groups": [],
                "research_gaps": [],
                "references": [{"id": "r1", "title": "Paper 1"}],
            },
        }
        passed, issues = agent.validate_output(output)
        assert passed is True
        assert len(issues) == 0

    def test_validates_abstract_writing_success(self):
        agent = PaperWritingAgent()
        output = {
            "status": "completed",
            "task_id": "T-1",
            "task_type": TASK_ABSTRACT_WRITING,
            "abstract": (
                "The system is presented. The results demonstrate a 15% "
                "improvement. This work provides theoretical insights."
            ),
            "sections": {
                "background": "bg",
                "motivation": "mot",
                "contribution": "contrib",
                "methodology": "method",
                "results": "results with 15% improvement",
                "significance": "sig",
            },
            "word_count": 20,
        }
        passed, issues = agent.validate_output(output)
        assert passed is True

    def test_validates_format_proofing_success(self):
        agent = PaperWritingAgent()
        output = {
            "status": "completed",
            "task_id": "T-1",
            "task_type": TASK_FORMAT_PROOFING,
            "original_text": "Original",
            "corrected_text": "Corrected",
            "issues_found": [],
        }
        passed, issues = agent.validate_output(output)
        assert passed is True

    def test_rejects_empty_output(self):
        agent = PaperWritingAgent()
        passed, issues = agent.validate_output({})
        assert passed is False
        assert len(issues) > 0

    def test_rejects_non_dict_output(self):
        agent = PaperWritingAgent()
        passed, issues = agent.validate_output("not a dict")
        assert passed is False

    def test_detects_missing_status(self):
        agent = PaperWritingAgent()
        output = {"task_id": "T-1", "task_type": TASK_FORMAT_PROOFING}
        passed, issues = agent.validate_output(output)
        assert any("status" in i.lower() for i in issues)

    def test_detects_missing_task_id(self):
        agent = PaperWritingAgent()
        output = {"status": "completed", "task_type": TASK_FORMAT_PROOFING}
        passed, issues = agent.validate_output(output)
        assert any("task_id" in i.lower() for i in issues)

    def test_detects_empty_references_in_review(self):
        agent = PaperWritingAgent()
        output = {
            "status": "completed",
            "task_id": "T-1",
            "task_type": TASK_LITERATURE_REVIEW,
            "sections": {
                "introduction": "Intro",
                "thematic_groups": [],
                "research_gaps": [],
                "references": [],
            },
        }
        passed, issues = agent.validate_output(output)
        assert any("empty" in i.lower() for i in issues)


class TestTaskTypeResolution:
    def test_resolves_from_metadata(self):
        agent = PaperWritingAgent()
        task = DomainTask(
            task_id="T-1",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Unknown Name",
            objective="Unknown objective",
            metadata={"task_type": TASK_FORMAT_PROOFING},
        )
        resolved = agent._resolve_task_type(task)
        assert resolved == TASK_FORMAT_PROOFING

    def test_resolves_from_task_name_literature(self):
        agent = PaperWritingAgent()
        task = DomainTask(
            task_id="T-1",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Literature Review Task",
            objective="Do something",
        )
        resolved = agent._resolve_task_type(task)
        assert resolved == TASK_LITERATURE_REVIEW

    def test_resolves_from_task_name_abstract(self):
        agent = PaperWritingAgent()
        task = DomainTask(
            task_id="T-1",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Write Abstract",
            objective="Do something",
        )
        resolved = agent._resolve_task_type(task)
        assert resolved == TASK_ABSTRACT_WRITING

    def test_resolves_from_task_name_format(self):
        agent = PaperWritingAgent()
        task = DomainTask(
            task_id="T-1",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Format Proofing Check",
            objective="Do something",
        )
        resolved = agent._resolve_task_type(task)
        assert resolved == TASK_FORMAT_PROOFING

    def test_defaults_to_literature_review(self):
        agent = PaperWritingAgent()
        task = DomainTask(
            task_id="T-1",
            domain_type=DomainType.PAPER_WRITING,
            task_name="Some Random Task",
            objective="Random objective",
        )
        resolved = agent._resolve_task_type(task)
        assert resolved == TASK_LITERATURE_REVIEW


class TestCalibration:
    def test_calibration_runs_all_tasks(self):
        agent = PaperWritingAgent()
        results = agent.calibrate()
        assert len(results) >= 3

    def test_calibration_updates_profile(self):
        agent = PaperWritingAgent()
        agent.calibrate()
        assert agent.profile.sample_count > 0

    def test_calibration_status_after_run(self):
        agent = PaperWritingAgent()
        agent.calibrate()
        assert agent.profile.calibration_status in {
            CalibrationStatus.CALIBRATED,
            CalibrationStatus.DEGRADED,
        }

    def test_calibration_summary_available(self):
        agent = PaperWritingAgent()
        agent.calibrate()
        summary = agent.calibration_summary()
        assert summary["total_runs"] >= 3


class TestDomainRegistryIntegration:
    def test_register_paper_writing_agent(self):
        registry = DomainRegistry()
        agent = PaperWritingAgent("pw_1")
        registry.register(agent)
        assert registry.agent_count == 1

    def test_find_by_domain(self):
        registry = DomainRegistry()
        agent = PaperWritingAgent("pw_1")
        registry.register(agent)
        agents = registry.get_agents_by_domain(DomainType.PAPER_WRITING)
        assert len(agents) == 1
        assert agents[0].agent_id == "pw_1"

    def test_find_best_agent_for_paper_writing_skill(self):
        registry = DomainRegistry()
        agent = PaperWritingAgent("pw_1")
        agent.profile.skills["literature_review"] = 0.9
        registry.register(agent)
        best = registry.find_best_agent("literature_review")
        assert best is not None
        assert best.agent_id == "pw_1"


class TestPassiveVoiceDetection:
    def test_passive_voice_detected(self):
        agent = PaperWritingAgent()
        text = "The structure is presented. Experiments are performed."
        ratio = agent._check_passive_voice_ratio(text)
        assert ratio > 0.0

    def test_no_passive_voice(self):
        agent = PaperWritingAgent()
        text = "We propose a new method. I design the system."
        ratio = agent._check_passive_voice_ratio(text)
        assert ratio == 0.0

    def test_empty_text(self):
        agent = PaperWritingAgent()
        ratio = agent._check_passive_voice_ratio("")
        assert ratio == 0.0


class TestWordCount:
    def test_basic_word_count(self):
        agent = PaperWritingAgent()
        count = agent._count_words("hello world foo bar")
        assert count == 4

    def test_empty_string(self):
        agent = PaperWritingAgent()
        count = agent._count_words("")
        assert count == 0
