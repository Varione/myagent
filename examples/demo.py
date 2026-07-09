"""
Symposium \u6f14\u793a\u811a\u672c

\u6a21\u62df\u4e00\u4e2a\u201c\u591a\u6a21\u578b\u7814\u8ba8 + \u534f\u540c\u6267\u884c\u201d\u7684\u5b8c\u6574\u6d41\u7a0b\u3002
\u4f7f\u7528 MockModel \u6a21\u62df\u4e0d\u540c\u4e13\u5bb6\u89d2\u8272\uff0c\u65e0\u9700\u771f\u5b9e API\u3002

\u8fd0\u884c: python examples/demo.py
"""

import sys
import os
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from symposium.models.base import BaseModel, ChatMessage, ModelResponse
from symposium.config import SymposiumConfig, DeliberationConfig, ExecutionConfig
from symposium.core.workflow import SymposiumWorkflow


_call_counters: dict = {}


class MockModel(BaseModel):
    """\u6a21\u62df\u6a21\u578b"""

    def __init__(self, name: str, persona: str, style: str, weight: float = 1.0):
        self._name = name
        self._persona = persona
        self._style = style
        self._weight = weight
        _call_counters[name] = 0

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages: list[ChatMessage], **kwargs) -> ModelResponse:
        _call_counters[self._name] += 1
        call_num = _call_counters[self._name]
        content = self._pick_response(call_num)
        return ModelResponse(content=content, model_name=self._name)

    def _pick_response(self, call_num: int) -> str:
        # Main model (oracle) first call = synthesis
        if self._name == "oracle":
            if call_num == 1:
                return self._synthesis_response()
            else:
                return self._execution_output()

        # Debate models: route by name, not call_num
        if call_num == 1:
            mapping = {
                "architect": self._first_round_architect,
                "engineer": self._first_round_engineer,
                "reviewer": self._first_round_reviewer,
            }
            return mapping.get(self._name, self._first_round_architect)()
        elif call_num == 2:
            mapping = {
                "architect": self._second_round_architect,
                "engineer": self._second_round_engineer,
                "reviewer": self._second_round_reviewer,
            }
            return mapping.get(self._name, self._second_round_architect)()
        else:
            return self._execution_output()

    def _first_round_architect(self) -> str:
        return (
            "\u4ece\u67b6\u6784\u89c6\u89d2\u6765\u770b\uff0c\u6211\u8ba4\u4e3a\u8fd9\u4e2a\u4efb\u52a1\u7684\u6838\u5fc3\u5728\u4e8e\u4ee5\u4e0b\u4e09\u70b9\uff1a\n\n"
            "1. **\u5206\u5c42\u8bbe\u8ba1**\uff1a\u7cfb\u7edf\u5e94\u8be5\u91c7\u7528\u6e05\u6670\u7684\u5206\u5c42\u67b6\u6784\uff0c\u5c06\u5173\u6ce8\u70b9\u5206\u79bb\u3002"
            "\u5efa\u8bae\u5206\u4e3a\u63a5\u53e3\u5c42\u3001\u4e1a\u52a1\u903b\u8f91\u5c42\u548c\u6570\u636e\u5c42\uff0c\u6bcf\u5c42\u804c\u8d23\u5355\u4e00\u3002\n\n"
            "2. **\u53ef\u6269\u5c55\u6027**\uff1a\u8bbe\u8ba1\u65b9\u6848\u65f6\u5fc5\u987b\u8003\u8651\u672a\u6765\u7684\u6269\u5c55\u9700\u6c42\u3002"
            "\u5efa\u8bae\u91c7\u7528\u63d2\u4ef6\u5316\u67b6\u6784\uff0c\u6838\u5fc3\u529f\u80fd\u4e0e\u6269\u5c55\u529f\u80fd\u89e3\u8026\u3002\n\n"
            "3. **\u6a21\u5757\u95f4\u901a\u4fe1**\uff1a\u6a21\u5757\u95f4\u901a\u4fe1\u534f\u8bae\u9700\u8981\u660e\u786e\u5b9a\u4e49\uff0c"
            "\u5efa\u8bae\u91c7\u7528\u4e8b\u4ef6\u9a71\u52a8\u67b6\u6784\uff0c\u964d\u4f4e\u6a21\u5757\u95f4\u8026\u5408\u5ea6\u3002"
        )

    def _first_round_engineer(self) -> str:
        return (
            "\u6211\u4ece\u843d\u5730\u89d2\u5ea6\u5206\u6790\u8fd9\u4e2a\u4efb\u52a1\uff0c\u5173\u6ce8\u70b9\u5728\u4e8e\uff1a\n\n"
            "1. **\u6280\u672f\u6808\u9009\u62e9**\uff1a\u9700\u8981\u8003\u8651\u56e2\u961f\u719f\u6089\u5ea6\u548c\u751f\u6001\u6210\u719f\u5ea6\u3002"
            "\u5efa\u8bae\u4f18\u5148\u9009\u62e9\u793e\u533a\u6d3b\u8dc3\u3001\u6587\u6863\u5b8c\u5584\u7684\u6280\u672f\u65b9\u6848\u3002\n\n"
            "2. **\u5b9e\u73b0\u590d\u6742\u5ea6**\uff1a\u67b6\u6784\u5e08\u7684\u65b9\u6848\u5728\u7406\u8bba\u4e0a\u5f88\u5b8c\u7f8e\uff0c"
            "\u4f46\u5b9e\u73b0\u4e2d\u4f1a\u6709\u51e0\u4e2a\u96be\u70b9\u9700\u8981\u8003\u8651\uff1a\n"
            "   - \u63d2\u4ef6\u5316\u67b6\u6784\u7684\u63a5\u53e3\u7a33\u5b9a\u6027\u95ee\u9898\n"
            "   - \u4e8b\u4ef6\u9a71\u52a8\u5e26\u6765\u7684\u8c03\u8bd5\u590d\u6742\u5ea6\n"
            "   - \u5206\u5e03\u5f0f\u4e00\u81f4\u6027\u7684\u4fdd\u969c\n\n"
            "3. **\u5de5\u7a0b\u6210\u672c**\uff1a\u5efa\u8bae\u5206\u9636\u6bb5\u5b9e\u65bd\uff0cMVP \u5148\u9a8c\u8bc1\u6838\u5fc3\u6d41\u7a0b\uff0c"
            "\u518d\u9010\u6b65\u6dfb\u52a0\u9ad8\u7ea7\u7279\u6027\u3002\u524d\u671f\u8fc7\u5ea6\u8bbe\u8ba1\u4f1a\u589e\u52a0\u4ea4\u4ed8\u98ce\u9669\u3002"
        )

    def _first_round_reviewer(self) -> str:
        return (
            "\u6211\u4ece\u98ce\u9669\u5ba1\u67e5\u89d2\u5ea6\u63d0\u51fa\u51e0\u70b9\u5173\u6ce8\uff1a\n\n"
            "1. **\u8fb9\u754c\u60c5\u51b5**\uff1a\u4e0a\u9762\u7684\u65b9\u6848\u90fd\u6ca1\u6709\u8003\u8651\u5f02\u5e38\u573a\u666f\u3002"
            "\u6bd4\u5982\u7f51\u7edc\u5206\u533a\u3001\u8282\u70b9\u5b95\u673a\u3001\u6d88\u606f\u4e22\u5931\u7b49\u60c5\u51b5\u4e0b\u7cfb\u7edf\u884c\u4e3a\u5982\u4f55\uff1f\n\n"
            "2. **\u5b89\u5168\u6027**\uff1a\u6a21\u5757\u95f4\u901a\u4fe1\u662f\u5426\u9700\u8981\u52a0\u5bc6\uff1f"
            "\u6570\u636e\u8bbf\u95ee\u6743\u9650\u5982\u4f55\u63a7\u5236\uff1f\u7f3a\u5c11\u5bf9\u8fd9\u4e9b\u975e\u529f\u80fd\u6027\u9700\u6c42\u7684\u8ba8\u8bba\u3002\n\n"
            "3. **\u53ef\u89c2\u6d4b\u6027**\uff1a\u5206\u5e03\u5f0f\u7cfb\u7edf\u5fc5\u987b\u8003\u8651\u76d1\u63a7\u3001\u65e5\u5fd7\u3001\u8ffd\u8e2a\u3002"
            "\u65b9\u6848\u4e2d\u6ca1\u6709\u63d0\u53ca\u5982\u4f55\u5b9a\u4f4d\u95ee\u9898\u548c\u76d1\u63a7\u7cfb\u7edf\u5065\u5eb7\u72b6\u6001\u3002\n\n"
            "4. **\u56de\u9000\u673a\u5236**\uff1a\u5982\u679c\u67d0\u4e2a\u6a21\u5757\u5347\u7ea7\u540e\u51fa\u73b0\u95ee\u9898\uff0c"
            "\u5982\u4f55\u5feb\u901f\u56de\u9000\uff1f\u9700\u8981\u8bbe\u8ba1\u7248\u672c\u517c\u5bb9\u548c\u7070\u5ea6\u53d1\u5e03\u7b56\u7565\u3002"
        )

    def _second_round_architect(self) -> str:
        return (
            "\u9605\u8bfb\u4e86\u524d\u9762\u5404\u4f4d\u7684\u89c2\u70b9\uff0c\u6211\u6709\u4ee5\u4e0b\u8865\u5145\u548c\u56de\u5e94\uff1a\n\n"
            "**\u540c\u610f\u5b9e\u73b0\u8005\u7684\u5206\u9636\u6bb5\u5efa\u8bae**\uff0c\u4f46\u60f3\u8865\u5145\u4e00\u70b9\uff1a"
            "\u67b6\u6784\u8bbe\u8ba1\u5e94\u8be5\u5728\u7b2c\u4e00\u9636\u6bb5\u5c31\u8003\u8651\u5230\u672a\u6765\u7684\u6269\u5c55\u70b9\uff0c"
            "\u800c\u4e0d\u662f\u5728\u6bcf\u4e2a\u9636\u6bb5\u91cd\u65b0\u8bbe\u8ba1\u3002\u5efa\u8bae\u91c7\u7528\u201c\u6f14\u8fdb\u5f0f\u67b6\u6784\u201d\u7684\u601d\u8def\uff0c"
            "\u5728\u521d\u671f\u5b9a\u4e49\u597d\u6269\u5c55\u63a5\u53e3\uff0c\u4f46\u4e0d\u8981\u8fc7\u5ea6\u5b9e\u73b0\u3002\n\n"
            "**\u5173\u4e8e\u5ba1\u67e5\u8005\u63d0\u51fa\u7684\u975e\u529f\u80fd\u6027\u9700\u6c42**\uff0c\u5b8c\u5168\u540c\u610f\u8fd9\u4e9b\u5f88\u91cd\u8981\u3002"
            "\u6211\u5efa\u8bae\u53ef\u89c2\u6d4b\u6027\u548c\u5b89\u5168\u6027\u4f5c\u4e3a\u67b6\u6784\u7684\u8de8\u5207\u9762\u5173\u6ce8\u70b9\uff0c"
            "\u8d2f\u7a7f\u6240\u6709\u5c42\uff0c\u800c\u4e0d\u662f\u4f5c\u4e3a\u5355\u72ec\u6a21\u5757\u3002\n\n"
            "**\u8865\u5145\u4e00\u4e2a\u89c2\u70b9**\uff1a\u5bf9\u4e8e\u6a21\u5757\u95f4\u901a\u4fe1\uff0c\u5efa\u8bae\u660e\u786e\u540c\u6b65\u8c03\u7528\u548c\u5f02\u6b65\u6d88\u606f\u7684\u8fb9\u754c\u3002"
            "\u67e5\u8be2\u7c7b\u64cd\u4f5c\u9002\u5408\u540c\u6b65\u63a5\u53e3\uff0c\u547d\u4ee4\u7c7b\u64cd\u4f5c\u9002\u5408\u5f02\u6b65\u4e8b\u4ef6\u3002"
        )

    def _second_round_engineer(self) -> str:
        return (
            "\u7efc\u5408\u5927\u5bb6\u7684\u610f\u89c1\uff0c\u6211\u6765\u8bc4\u4f30\u5b9e\u73b0\u5c42\u9762\uff1a\n\n"
            "**\u5173\u4e8e\u67b6\u6784\u5e08\u63d0\u5230\u7684\u6f14\u8fdb\u5f0f\u67b6\u6784**\uff0c\u8fd9\u4e2a\u601d\u8def\u53ef\u884c\u3002"
            "\u5b9e\u8df5\u4e2d\u6211\u4eec\u53ef\u4ee5\u91c7\u7528\u4ee5\u4e0b\u7b56\u7565\uff1a\n"
            "1. \u7b2c\u4e00\u671f\uff1a\u5355\u4f53\u67b6\u6784 + \u6e05\u6670\u7684\u6a21\u5757\u8fb9\u754c\n"
            "2. \u7b2c\u4e8c\u671f\uff1a\u6309\u4e1a\u52a1\u57df\u62c6\u5206\u72ec\u7acb\u670d\u52a1\n"
            "3. \u7b2c\u4e09\u671f\uff1a\u5f15\u5165\u4e8b\u4ef6\u603b\u7ebf\n\n"
            "**\u5173\u4e8e\u5ba1\u67e5\u8005\u7684\u98ce\u9669\u70b9**\uff0c\u6211\u8ba4\u4e3a\u5b89\u5168\u6027\u53ef\u4ee5\u5206\u9636\u6bb5\u5b9e\u65bd\uff1a"
            "\u521d\u671f\u4f7f\u7528 mTLS \u8fdb\u884c\u670d\u52a1\u95f4\u8ba4\u8bc1\uff0c"
            "\u540e\u671f\u518d\u5f15\u5165\u66f4\u7ec6\u7c92\u5ea6\u7684\u6743\u9650\u63a7\u5236\u3002\n\n"
            "**\u6280\u672f\u9009\u578b\u5efa\u8bae**\uff1a\u4f18\u5148\u9009\u62e9\u540c\u65f6\u652f\u6301\u540c\u6b65\u548c\u5f02\u6b65\u901a\u4fe1\u7684\u6846\u67b6\uff0c"
            "\u964d\u4f4e\u540e\u671f\u7684\u8fc1\u79fb\u6210\u672c\u3002\u63a8\u8350\u4f7f\u7528 gRPC \u505a\u540c\u6b65\u8c03\u7528\uff0c"
            "\u914d\u5408\u6d88\u606f\u961f\u5217\u505a\u5f02\u6b65\u4e8b\u4ef6\u3002"
        )

    def _second_round_reviewer(self) -> str:
        return (
            "\u9488\u5bf9\u65b0\u7684\u65b9\u6848\uff0c\u6211\u8fdb\u884c\u7b2c\u4e8c\u8f6e\u5ba1\u67e5\uff1a\n\n"
            "**\u6b63\u9762**\uff1a\u6f14\u8fdb\u5f0f\u67b6\u6784\u7684\u601d\u8def\u5f88\u597d\uff0c\u964d\u4f4e\u4e86\u8fc7\u5ea6\u8bbe\u8ba1\u98ce\u9669\u3002"
            "\u5206\u9636\u6bb5\u8ba1\u5212\u4e5f\u5207\u5408\u5b9e\u9645\u3002\n\n"
            "**\u5269\u4f59\u98ce\u9669**\uff1a\n"
            "1. gRPC + \u6d88\u606f\u961f\u5217\u7684\u53cc\u901a\u4fe1\u6a21\u5f0f\u589e\u52a0\u4e86\u8fd0\u7ef4\u590d\u6742\u5ea6\uff0c"
            "\u56e2\u961f\u662f\u5426\u5177\u5907\u76f8\u5e94\u80fd\u529b\uff1f\n"
            "2. \u7b2c\u4e00\u671f\u5230\u7b2c\u4e8c\u671f\u7684\u67b6\u6784\u8fc1\u79fb\u5982\u4f55\u5e73\u6ed1\u8fdb\u884c\uff1f"
            "\u6570\u636e\u8fc1\u79fb\u65b9\u6848\u9700\u8981\u63d0\u524d\u8bbe\u8ba1\u3002\n"
            "3. \u6280\u672f\u9009\u578b\u7f3a\u4e4f\u5907\u9009\u65b9\u6848\uff0c\u5982\u679c\u6240\u9009\u6846\u67b6\u4e0d\u6ee1\u8db3\u9700\u6c42\uff0c\u66ff\u4ee3\u65b9\u6848\u662f\u4ec0\u4e48\uff1f\n\n"
            "**\u5efa\u8bae**\uff1a\u5728\u8fdb\u5165\u5b9e\u73b0\u524d\uff0c\u81f3\u5c11\u5b8c\u6210\u4e00\u4e2a\u5173\u952e\u8def\u5f84\u7684\u7aef\u5230\u7aef\u6280\u672f\u539f\u578b\uff0c"
            "\u9a8c\u8bc1\u6280\u672f\u9009\u578b\u7684\u53ef\u884c\u6027\u3002"
        )

    def _synthesis_response(self) -> str:
        return (
            "## \u5171\u8bc6\u5206\u6790\n"
            "- \u6240\u6709\u6a21\u578b\u5747\u8ba4\u4e3a\u9700\u8981\u91c7\u7528\u5206\u5c42\u67b6\u6784\uff0c\u5c06\u5173\u6ce8\u70b9\u5206\u79bb\n"
            "- \u4e00\u81f4\u540c\u610f\u9700\u8981\u5206\u9636\u6bb5\u5b9e\u65bd\uff08MVP \u5148\u884c\uff09\n"
            "- \u5747\u8ba4\u4e3a\u975e\u529f\u80fd\u6027\u9700\u6c42\uff08\u53ef\u89c2\u6d4b\u6027\u3001\u5b89\u5168\u6027\uff09\u81f3\u5173\u91cd\u8981\n\n"
            "## \u5206\u6b67\u4e0e\u88c1\u51b3\n"
            "- \u67b6\u6784\u65b9\u6848\u7ec6\u8282: \u67b6\u6784\u5e08\u503e\u5411\u4e8b\u4ef6\u9a71\u52a8\uff0c\u5b9e\u73b0\u8005\u5efa\u8bae gRPC+\u6d88\u606f\u961f\u5217 "
            "-> \u88c1\u51b3: \u91c7\u7528\u6df7\u5408\u6a21\u5f0f\uff0c\u67e5\u8be2\u7528 gRPC\uff0c\u4e8b\u4ef6\u7528\u6d88\u606f\u961f\u5217\n"
            "- \u63d2\u4ef6\u5316\u7ea7\u522b: \u67b6\u6784\u5e08\u5efa\u8bae\u5f3a\u63d2\u4ef6\u5316\uff0c\u5ba1\u67e5\u8005\u62c5\u5fc3\u590d\u6742\u5ea6 "
            "-> \u88c1\u51b3: MVP \u671f\u4e0d\u7528\u63d2\u4ef6\u4f53\u7cfb\uff0c\u7b2c\u4e8c\u671f\u518d\u5f15\u5165\n\n"
            "## \u6267\u884c\u8ba1\u5212\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "title": "\u67b6\u6784\u8bbe\u8ba1\u6587\u6863",\n'
            '    "description": "\u8f93\u51fa\u7cfb\u7edf\u6574\u4f53\u67b6\u6784\u8bbe\u8ba1\u6587\u6863\uff0c\u5305\u542b\u5206\u5c42\u67b6\u6784\u3001\u6a21\u5757\u8fb9\u754c\u548c\u63a5\u53e3\u534f\u8bae\u5b9a\u4e49",\n'
            '    "assigned_model": "architect",\n'
            '    "dependencies": []\n'
            "  },\n"
            "  {\n"
            '    "title": "\u6280\u672f\u9009\u578b\u8bc4\u4f30\u62a5\u544a",\n'
            '    "description": "\u8bc4\u4f30 gRPC\u3001\u6d88\u606f\u961f\u5217\u3001\u63d2\u4ef6\u6846\u67b6\u7b49\u6280\u672f\u9009\u578b\uff0c\u8f93\u51fa\u5bf9\u6bd4\u5206\u6790\u548c\u63a8\u8350\u65b9\u6848",\n'
            '    "assigned_model": "engineer",\n'
            '    "dependencies": []\n'
            "  },\n"
            "  {\n"
            '    "title": "\u975e\u529f\u80fd\u6027\u9700\u6c42\u6e05\u5355",\n'
            '    "description": "\u68c0\u67e5\u5e76\u8865\u5145\u7cfb\u7edf\u7684\u5b89\u5168\u6027\u3001\u53ef\u89c2\u6d4b\u6027\u3001\u53ef\u9760\u6027\u9700\u6c42\uff0c\u8f93\u51fa\u975e\u529f\u80fd\u6027\u9700\u6c42\u6e05\u5355",\n'
            '    "assigned_model": "reviewer",\n'
            '    "dependencies": []\n'
            "  }\n"
            "]\n"
            "```"
        )

    def _execution_output(self) -> str:
        return (
            f"\u3010{self._name} \u6267\u884c\u7ed3\u679c\u3011\n\n"
            "\u5df2\u5b8c\u6210\u5206\u914d\u7684\u5b50\u4efb\u52a1\u3002\u4ee5\u4e0b\u662f\u4ea4\u4ed8\u7269\uff1a\n\n"
            "```python\n"
            "from abc import ABC, abstractmethod\n"
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Result:\n"
            '    """Generic result wrapper"""\n'
            "    success: bool\n"
            "    data: any = None\n"
            "    error: str = ''\n\n\n"
            "class ModuleInterface(ABC):\n"
            '    """Module interface contract"""\n\n'
            "    @abstractmethod\n"
            "    async def initialize(self) -> bool:\n"
            "        ...\n\n"
            "    @abstractmethod\n"
            "    async def handle(self, request: dict) -> Result:\n"
            "        ...\n"
            "```\n\n"
            "\u4ee5\u4e0a\u4ee3\u7801\u5df2\u901a\u8fc7\u57fa\u672c\u8bed\u6cd5\u68c0\u67e5\u3002"
        )


def create_mock_models() -> dict:
    return {
        "architect": MockModel(
            name="architect",
            persona="\u67b6\u6784\u5e08 - \u64c5\u957f\u4ece\u5168\u5c40\u89c6\u89d2\u5206\u6790\u95ee\u9898",
            style="architect",
        ),
        "engineer": MockModel(
            name="engineer",
            persona="\u5b9e\u73b0\u8005 - \u5173\u6ce8\u5177\u4f53\u843d\u5730\u7ec6\u8282",
            style="implementer",
        ),
        "reviewer": MockModel(
            name="reviewer",
            persona="\u5ba1\u67e5\u8005 - \u64c5\u957f\u53d1\u73b0\u98ce\u9669\u548c\u6f0f\u6d1e",
            style="reviewer",
        ),
        "oracle": MockModel(
            name="oracle",
            persona="\u4e3b\u6a21\u578b - \u7efc\u5408\u51b3\u7b56\u8005",
            style="synthesizer",
            weight=1.2,
        ),
    }


def run_demo(task_description: str, task_context: str = ""):
    print("\n" + "=" * 70)
    print("  Symposium \u591a\u6a21\u578b\u534f\u540c Agent \u6f14\u793a")
    print("=" * 70)

    config = SymposiumConfig()
    config.deliberation = DeliberationConfig(num_rounds=2)
    config.execution = ExecutionConfig(max_concurrent_tasks=2)
    config.main_model = "oracle"

    models = create_mock_models()
    workflow = SymposiumWorkflow(config, models=models)

    print(f"\n[*] \u4efb\u52a1: {task_description}")
    print(f"[M] \u53c2\u4e0e\u6a21\u578b: {', '.join(models.keys())}")
    print(f"[O] \u4e3b\u6a21\u578b: oracle")
    print(f"[R] \u7814\u8ba8\u8f6e\u6570: 2")
    print(f"\n{'─' * 70}")
    print("\u5f00\u59cb\u7814\u8ba8\u9636\u6bb5...\n")

    print("\n" + "=" * 70)
    print("  \u9636\u6bb5\u4e00: \u5706\u684c\u7814\u8ba8 (Roundtable Deliberation)")
    print("=" * 70)

    result = workflow.run(task_description, task_context)

    workflow.print_report(result)

    print("\n" + "=" * 70)
    print("  \u6f14\u793a\u7ed3\u8bba")
    print("=" * 70)
    print(f"  \u603b\u8017\u65f6: {result['total_time']:.1f}s")
    print(f"  [OK] \u7814\u8ba8\u53d1\u8a00: {len(result['deliberation'].contributions)} \u6b21")
    print(f"  [OK] \u4efb\u52a1\u5206\u89e3: {len(result['plan'].tasks)} \u4e2a")
    successful = sum(1 for t in result['plan'].tasks if t.status.value == 'done')
    print(f"  [OK] \u6267\u884c\u5b8c\u6210: {successful}/{len(result['plan'].tasks)}")

    print(f"\n{'═' * 70}")
    print(f"  \u6f14\u793a\u7ed3\u675f - Symposium \u591a\u6a21\u578b\u534f\u540c\u67b6\u6784\u9a8c\u8bc1\u901a\u8fc7")
    print(f"{'═' * 70}\n")

    return result


if __name__ == "__main__":
    task = (
        "\u8bbe\u8ba1\u4e00\u4e2a\u5fae\u670d\u52a1\u67b6\u6784\u7684\u5b9e\u65f6\u6570\u636e\u5904\u7406\u5e73\u53f0\uff0c"
        "\u9700\u8981\u652f\u6301\u9ad8\u541e\u5410\u91cf\u6570\u636e\u63a5\u5165\u3001\u5b9e\u65f6\u8ba1\u7b97\u548c\u53ef\u89c6\u5316\u5c55\u793a\u3002"
        "\u7cfb\u7edf\u8981\u6c42\u9ad8\u53ef\u7528\u3001\u53ef\u6c34\u5e73\u6269\u5c55\uff0c\u5e76\u652f\u6301\u591a\u79df\u6237\u9694\u79bb\u3002"
    )

    context = (
        "\u76ee\u6807\u573a\u666f: IoT \u8bbe\u5907\u6570\u636e\u91c7\u96c6\u4e0e\u5b9e\u65f6\u5206\u6790\n"
        "\u9884\u671f\u541e\u5410: 10\u4e07\u6761/\u79d2\n"
        "\u53ef\u7528\u8d44\u6e90: 8 \u8282\u70b9 Kubernetes \u96c6\u7fa4\n"
        "\u65f6\u95f4\u8981\u6c42: 3 \u4e2a\u6708\u4ea4\u4ed8 MVP"
    )

    run_demo(task, context)
