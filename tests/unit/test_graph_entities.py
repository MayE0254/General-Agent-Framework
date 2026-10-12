"""Unit tests for the LLM entity layer: vocabulary, parsing, and degradation.

No network and no Neo4j: the LLM boundary is ``LLMService`` with an injected
``completion_fn``, which is exactly the seam the production code uses.  The
degradation contract is tested by making the primary leg fail the way a quota
does, because that path decides whether a multi-hour run keeps going or throws
away everything it has already paid for.

The clause strings are corpus excerpts (企税法第四条 / 第九十一条 / 个税法第六条),
so a "does the vocabulary cover what the corpus says" regression fails here
rather than after a full paid run.
"""

import asyncio
import json

from app.core.settings import GraphEntitySettings, LLMSettings
from app.knowledge.graph.entities import (
    ENTITY_TYPE_KEYS,
    MAX_ENTITIES_PER_CLAUSE,
    ClauseInput,
    EntityParseError,
    assign_refs,
    build_messages,
    entity_id,
    is_truncated,
    normalize_entity_name,
    parse_extraction,
    resolve_type_key,
    split_batch,
)
from app.knowledge.graph.entity_client import (
    EntityExtractionError,
    EntityExtractor,
    FailureKind,
    classify_failure,
)
from app.llm.service import LLMService

# --------------------------------------------------------------------------
# Corpus excerpts
# --------------------------------------------------------------------------
CLAUSE_CIT_4 = ClauseInput(
    clause_id="clause-4",
    doc_id="doc:qsf",
    title="中华人民共和国企业所得税法",
    section="第四条",
    text="企业所得税的税率为25%。非居民企业取得本法第三条第三款规定的所得，适用税率为20%。",
)
CLAUSE_CIT_91 = ClauseInput(
    clause_id="clause-91",
    doc_id="doc:qsf-tiaoli",
    title="中华人民共和国企业所得税法实施条例",
    section="第九十一条",
    text="非居民企业取得企业所得税法第二十七条第(五)项规定的所得，减按10%的税率征收企业所得税。",
)
CLAUSE_IIT_6 = ClauseInput(
    clause_id="clause-6",
    doc_id="doc:gsf",
    title="中华人民共和国个人所得税法",
    section="第六条",
    text="专项附加扣除包括子女教育、继续教育、大病医疗、住房贷款利息或者住房租金、赡养老人等支出。",
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt_tokens: int = 120, completion_tokens: int = 40) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class _Raw:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage()
        self.model = "glm-5.3"


def _answer_for(messages, entities=None) -> str:
    """Build a valid answer that echoes every ref found in the prompt.

    ``messages`` arrives through ``completion_fn``, i.e. after
    ``LLMService._build_kwargs`` has dumped every ``LLMMessage`` to a plain
    dict -- reading ``.content`` here would find nothing and quietly answer
    with an empty clause list.
    """
    import re

    last = messages[-1]
    user = last["content"] if isinstance(last, dict) else last.content
    refs = re.findall(r"【(c\d+)】", user or "")
    payload = []
    for ref in refs:
        payload.append({"ref": ref, "entities": entities or [{"type": "taxpayer", "name": "非居民企业"}]})
    return json.dumps({"clauses": payload}, ensure_ascii=False)


def _entity_settings(**overrides) -> GraphEntitySettings:
    base = dict(
        enabled=True,
        provider="zhipu",
        model="zai/glm-5.3",
        temperature=0.0,
        max_tokens=4096,
        timeout_seconds=30,
        concurrency=1,
        max_clauses_per_call=20,
        max_chars_per_call=12000,
        max_split_depth=3,
        fallback_provider="deepseek",
        fallback_model="deepseek/deepseek-v4-flash",
        fallback_extra_body={"thinking": {"type": "enabled"}},
    )
    base.update(overrides)
    return GraphEntitySettings(**base)


def _llm_settings() -> LLMSettings:
    return LLMSettings(
        api_key="sk-default",
        model="deepseek/deepseek-v4-flash",
        max_retries=0,
        providers={
            "zhipu": {
                "api_base": "https://open.bigmodel.cn/api/paas/v4",
                "api_key": "sk-zhipu",
            },
            "deepseek": {
                "api_base": "https://api.deepseek.com",
                "api_key": "sk-deepseek",
            },
        },
    )


class _Recorder:
    """Completion fake that records every request and can fail per model."""

    def __init__(self, fail_models: dict[str, list[str]] | None = None) -> None:
        self.calls: list[dict] = []
        # model -> queue of exception messages to raise, popped per call.
        self.fail_models = fail_models or {}

    async def __call__(self, kwargs: dict) -> object:
        self.calls.append(kwargs)
        queue = self.fail_models.get(kwargs["model"])
        if queue:
            raise RuntimeError(queue.pop(0))
        return _Raw(_answer_for(kwargs["messages"]))

    @property
    def models(self) -> list[str]:
        return [call["model"] for call in self.calls]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
def test_vocabulary_is_closed_and_has_no_catch_all() -> None:
    assert ENTITY_TYPE_KEYS == ("taxpayer", "tax_type", "tax_matter", "tax_base", "admin")
    assert [resolve_type_key(key) for key in ENTITY_TYPE_KEYS] == list(ENTITY_TYPE_KEYS)
    # The Chinese label is accepted as an answer too, and normalised to the key.
    assert resolve_type_key("纳税人主体") == "taxpayer"
    assert resolve_type_key(" 税种 ") == "tax_type"
    # Anything else is rejected: a miscellaneous bucket is where junk collects.
    assert resolve_type_key("other") is None
    assert resolve_type_key("") is None
    assert resolve_type_key("公司") is None


def test_normalize_entity_name_is_mechanical_only() -> None:
    assert normalize_entity_name(" 非居民企业 ") == "非居民企业"
    assert normalize_entity_name("《中华人民共和国个人所得税法》") == "中华人民共和国个人所得税法"
    assert normalize_entity_name("专项附加扣除等") == "专项附加扣除"
    assert normalize_entity_name("3岁以下婴幼儿照护。") == "3岁以下婴幼儿照护"
    # Synonyms are never folded: 居民企业 and 非居民企业 must stay distinct.
    assert normalize_entity_name("居民企业") != normalize_entity_name("非居民企业")


def test_entity_id_is_type_scoped() -> None:
    assert entity_id("tax_type", "个人所得税") == "tax_type:个人所得税"
    assert entity_id("tax_type", "个人所得税") != entity_id("tax_matter", "个人所得税")


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
def test_build_messages_carries_vocabulary_and_every_ref() -> None:
    ref_to_clause, rows = assign_refs([CLAUSE_CIT_4, CLAUSE_CIT_91])
    assert ref_to_clause == {"c1": "clause-4", "c2": "clause-91"}
    messages = build_messages("中华人民共和国企业所得税法", rows)
    system, user = messages[0].content, messages[1].content
    for key in ENTITY_TYPE_KEYS:
        assert key in system
    assert "非居民企业" in system  # the examples are what anchor the type list
    assert "【c1】第四条" in user and "【c2】第九十一条" in user
    assert CLAUSE_CIT_91.text in user
    assert str(MAX_ENTITIES_PER_CLAUSE) in system


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
_REF_MAP = {"c1": "clause-4", "c2": "clause-91"}


def test_parse_extraction_keeps_only_vocabulary_types() -> None:
    content = json.dumps(
        {
            "clauses": [
                {
                    "ref": "c1",
                    "entities": [
                        {"type": "taxpayer", "name": "非居民企业"},
                        {"type": "公司", "name": "某公司"},          # unknown type
                        {"type": "tax_base", "name": "所"},          # too short
                        {"type": "taxpayer", "name": "非居民企业"},  # duplicate
                    ],
                },
                {"ref": "c9", "entities": [{"type": "tax_type", "name": "企业所得税"}]},
                {"ref": "c2", "entities": [{"type": "tax_matter", "name": "减按10%的税率"}]},
            ]
        },
        ensure_ascii=False,
    )
    result = parse_extraction(content, _REF_MAP, "doc:qsf")
    assert [(m.clause_id, m.type_key, m.name) for m in result.mentions] == [
        ("clause-4", "taxpayer", "非居民企业"),
        ("clause-91", "tax_matter", "减按10%的税率"),
    ]
    assert result.stats["unknown_types"] == 1
    assert result.stats["invalid_names"] == 1
    assert result.stats["duplicates"] == 1
    assert result.stats["refs_unknown"] == 1
    assert result.stats["refs_returned"] == 2
    assert result.stats["refs_missing"] == 0
    assert result.stats["mentions"] == 2


def test_parse_extraction_counts_missing_refs() -> None:
    # The prompt requires every ref back, so an omitted one is measurable --
    # that is the only way to tell "no entities" from "the model got lazy".
    content = '{"clauses":[{"ref":"c1","entities":[]}]}'
    result = parse_extraction(content, _REF_MAP, "doc:qsf")
    assert result.stats["refs_missing"] == 1
    assert result.mentions == []


def test_same_entity_across_clauses_is_not_deduplicated() -> None:
    # Regression guard for the real defect of this round: the dedup key is
    # (clause_id, type_key, name), not (type_key, name).  A tax_type named
    # 企业所得税 mentioned in 第四条 and again in 第九十一条 is two mentions on
    # two clauses -- collapsing them silently detaches an entity from a clause
    # it actually appears in.
    content = json.dumps(
        {
            "clauses": [
                {"ref": "c1", "entities": [{"type": "tax_type", "name": "企业所得税"}]},
                {"ref": "c2", "entities": [{"type": "tax_type", "name": "企业所得税"}]},
            ]
        },
        ensure_ascii=False,
    )
    result = parse_extraction(content, _REF_MAP, "doc:qsf")
    assert [(m.clause_id, m.name) for m in result.mentions] == [
        ("clause-4", "企业所得税"),
        ("clause-91", "企业所得税"),
    ]
    assert result.stats["duplicates"] == 0
    assert result.stats["mentions"] == 2
    assert result.stats["entities"] == 1  # one node, reached from two clauses


def test_parse_extraction_tolerates_fences_prose_and_trailing_commas() -> None:
    fenced = '```json\n{"clauses":[{"ref":"c1","entities":[{"type":"tax_type","name":"企业所得税"}]}]}\n```'
    assert len(parse_extraction(fenced, _REF_MAP, "d").mentions) == 1
    prose = '好的，结果如下：\n{"clauses":[{"ref":"c1","entities":[]}]}\n以上。'
    assert parse_extraction(prose, _REF_MAP, "d").stats["refs_returned"] == 1
    trailing = '{"clauses":[{"ref":"c1","entities":[{"type":"admin","name":"纳税申报"},]},]}'
    assert parse_extraction(trailing, _REF_MAP, "d").mentions[0].name == "纳税申报"


def test_parse_extraction_accepts_the_chinese_label() -> None:
    content = '{"clauses":[{"ref":"c1","entities":[{"type":"征税对象","name":"豆制品"}]}]}'
    mention = parse_extraction(content, _REF_MAP, "d").mentions[0]
    assert (mention.type_key, mention.name) == ("tax_base", "豆制品")


def test_parse_extraction_raises_rather_than_returning_nothing() -> None:
    # An empty extraction is indistinguishable from "this document has no
    # entities", which would silently delete a document from the entity layer.
    for bad in ("", "not json at all", "[1, 2, 3]", '{"items": []}'):
        try:
            parse_extraction(bad, _REF_MAP, "d")
        except EntityParseError:
            continue
        raise AssertionError(f"expected EntityParseError for {bad!r}")


def test_is_truncated_detects_both_signals() -> None:
    assert is_truncated("length", '{"clauses": [') is True
    assert is_truncated("", '{"clauses": [{"ref": "c1"') is True
    assert is_truncated("stop", '{"clauses": []}') is False
    assert is_truncated("", "") is False


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
def test_split_batch_respects_both_caps_and_loses_nothing() -> None:
    many = [
        ClauseInput(clause_id=f"c{i}", doc_id="d", title="t", section=f"第{i}条", text="正文" * 10)
        for i in range(25)
    ]
    batches = split_batch(many, max_clauses=10, max_chars=10_000)
    assert [len(b) for b in batches] == [10, 10, 5]
    assert [c.clause_id for b in batches for c in b] == [c.clause_id for c in many]

    # A single clause larger than the char budget is kept whole, never dropped.
    huge = ClauseInput(clause_id="huge", doc_id="d", title="t", section="第一条", text="正" * 5000)
    batches = split_batch([huge], max_clauses=10, max_chars=100)
    assert [len(b) for b in batches] == [1]

    small = [CLAUSE_CIT_4, CLAUSE_CIT_91]
    assert split_batch(small, max_clauses=20, max_chars=50) == [[small[0]], [small[1]]]


def test_split_batch_of_empty_document_is_empty() -> None:
    assert split_batch([], max_clauses=10, max_chars=100) == []


# --------------------------------------------------------------------------
# Failure classification
# --------------------------------------------------------------------------
def test_classify_failure() -> None:
    assert classify_failure(RuntimeError("insufficient balance")) is FailureKind.QUOTA
    assert classify_failure(RuntimeError("账户余额不足")) is FailureKind.QUOTA
    assert classify_failure(RuntimeError("Error code: 429 - rate limit")) is FailureKind.RATE_LIMIT
    assert classify_failure(RuntimeError("并发超限")) is FailureKind.RATE_LIMIT
    assert classify_failure(RuntimeError("connection reset")) is FailureKind.OTHER


# --------------------------------------------------------------------------
# Orchestration: one call per document, degradation, provenance
# --------------------------------------------------------------------------
def _extractor(recorder: _Recorder, settings: GraphEntitySettings) -> EntityExtractor:
    service = LLMService(_llm_settings(), completion_fn=recorder)
    return EntityExtractor(service, settings)


def test_one_call_per_document_and_mentions_land_on_their_clause() -> None:
    recorder = _Recorder()
    extractor = _extractor(recorder, _entity_settings())
    result = asyncio.run(
        extractor.extract_document(
            "doc:qsf", "中华人民共和国企业所得税法", [CLAUSE_CIT_4, CLAUSE_CIT_91]
        )
    )
    assert len(recorder.calls) == 1  # two clauses, one document, one call
    assert {m.clause_id for m in result.mentions} == {"clause-4", "clause-91"}
    assert extractor.stats.calls == 1
    assert extractor.stats.primary_calls == 1
    assert extractor.stats.fallback_calls == 0
    assert extractor.stats.batches == 1
    # The primary leg must not receive the fallback's provider-specific fields.
    assert recorder.calls[0].get("extra_body") is None


def test_primary_leg_receives_its_own_extra_body() -> None:
    # GLM-5.3 always deliberates and its default effort blows through
    # max_tokens; the bounded effort is expressed as the primary leg's own
    # extra_body, so it must reach the request untouched.
    recorder = _Recorder()
    extractor = _extractor(
        recorder, _entity_settings(extra_body={"reasoning_effort": "low"})
    )
    asyncio.run(extractor.extract_document("doc:1", "t", [CLAUSE_CIT_4]))
    assert recorder.calls[0]["extra_body"] == {"reasoning_effort": "low"}
    # And it is not the fallback's field: the DeepSeek switch belongs to the
    # other leg and must not leak into the primary request.
    assert "thinking" not in recorder.calls[0]["extra_body"]


def test_quota_failure_degrades_and_opens_the_circuit() -> None:
    recorder = _Recorder(fail_models={"zai/glm-5.3": ["insufficient balance"]})
    extractor = _extractor(recorder, _entity_settings())

    first = asyncio.run(
        extractor.extract_document("doc:1", "doc one", [CLAUSE_CIT_4])
    )
    second = asyncio.run(
        extractor.extract_document("doc:2", "doc two", [CLAUSE_CIT_91])
    )

    assert first.mentions and second.mentions  # degraded, not lost
    assert recorder.models == [
        "zai/glm-5.3",              # 1st doc: primary tried once
        "deepseek/deepseek-v4-flash",  # 1st doc: fallback answered
        "deepseek/deepseek-v4-flash",  # 2nd doc: straight to the fallback
    ]
    assert extractor.circuit_open is True
    assert "quota" in extractor.stats.circuit_reason
    assert extractor.stats.circuit_docs == 1
    assert extractor.stats.primary_calls == 1
    assert extractor.stats.fallback_calls == 2
    # The thinking switch belongs to the fallback leg only.
    assert recorder.calls[1]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert recorder.calls[0].get("extra_body") is None


def test_provenance_records_the_model_that_answered() -> None:
    recorder = _Recorder(fail_models={"zai/glm-5.3": ["余额不足"]})
    extractor = _extractor(recorder, _entity_settings())
    result = asyncio.run(extractor.extract_document("doc:1", "t", [CLAUSE_CIT_4]))
    assert {m.extractor for m in result.mentions} == {"deepseek/deepseek-v4-flash"}


def test_consecutive_failures_open_the_circuit_without_quota_wording() -> None:
    # A provider error whose text we cannot classify must still stop hammering
    # the primary leg once it is clearly not coming back.
    recorder = _Recorder(fail_models={"zai/glm-5.3": ["boom"] * 5})
    extractor = _extractor(recorder, _entity_settings())
    for index in range(3):
        asyncio.run(extractor.extract_document(f"doc:{index}", "t", [CLAUSE_CIT_4]))
    assert extractor.circuit_open is True
    assert "consecutive" in extractor.stats.circuit_reason
    assert extractor.stats.primary_calls == 3
    assert extractor.stats.fallback_calls == 3


def test_exhausted_fallback_raises_instead_of_returning_nothing() -> None:
    recorder = _Recorder(
        fail_models={
            "zai/glm-5.3": ["insufficient balance"],
            "deepseek/deepseek-v4-flash": ["deepseek down"],
        }
    )
    extractor = _extractor(recorder, _entity_settings())
    try:
        asyncio.run(extractor.extract_document("doc:1", "t", [CLAUSE_CIT_4]))
    except EntityExtractionError as exc:
        assert "deepseek" in str(exc)
    else:
        raise AssertionError("both legs failed but no error was raised")


def test_without_a_fallback_the_first_failure_propagates() -> None:
    recorder = _Recorder(fail_models={"zai/glm-5.3": ["insufficient balance"]})
    extractor = _extractor(
        recorder,
        _entity_settings(fallback_provider="", fallback_model="", fallback_extra_body={}),
    )
    assert extractor.has_fallback is False
    try:
        asyncio.run(extractor.extract_document("doc:1", "t", [CLAUSE_CIT_4]))
    except EntityExtractionError:
        pass
    else:
        raise AssertionError("expected the primary failure to propagate")
    assert len(recorder.calls) == 1


def test_truncated_answer_splits_the_batch_and_retries() -> None:
    refs_seen: list[int] = []

    async def _completion(kwargs: dict) -> object:
        user = kwargs["messages"][-1]["content"]
        count = user.count("【c")
        refs_seen.append(count)
        if len(refs_seen) == 1:  # first call is cut off by max_tokens
            return _Raw('{"clauses": [{"ref": "c1", "entities": [', "length")
        return _Raw(_answer_for(kwargs["messages"]))

    settings = _entity_settings(max_clauses_per_call=4)
    extractor = EntityExtractor(
        LLMService(_llm_settings(), completion_fn=_completion), settings
    )
    clauses = [
        ClauseInput(clause_id=f"k{i}", doc_id="d", title="t", section=f"第{i}条", text="正文")
        for i in range(4)
    ]
    result = asyncio.run(extractor.extract_document("doc:1", "t", clauses))
    assert refs_seen == [4, 2, 2]  # one call, then two halves
    assert {m.clause_id for m in result.mentions} == {"k0", "k1", "k2", "k3"}
    assert extractor.stats.truncated == 1
    assert extractor.stats.split_retries == 1


def test_unparseable_single_clause_is_a_hard_failure() -> None:
    async def _completion(kwargs: dict) -> object:
        return _Raw("sorry, I cannot help with that")

    extractor = EntityExtractor(
        LLMService(_llm_settings(), completion_fn=_completion), _entity_settings()
    )
    try:
        asyncio.run(extractor.extract_document("doc:1", "t", [CLAUSE_CIT_4]))
    except EntityExtractionError as exc:
        assert "unparseable" in str(exc)
    else:
        raise AssertionError("expected EntityExtractionError")


def test_discards_are_counted_across_documents() -> None:
    class _Bad:
        async def __call__(self, kwargs: dict) -> object:
            return _Raw('{"clauses":[{"ref":"c1","entities":[{"type":"公司","name":"某公司"}]}]}')

    extractor = EntityExtractor(LLMService(_llm_settings(), completion_fn=_Bad()), _entity_settings())
    result = asyncio.run(extractor.extract_document("doc:1", "t", [CLAUSE_CIT_4]))
    assert result.mentions == []
    assert extractor.stats.discards["unknown_types"] == 1
    assert extractor.stats.discards["mentions"] == 0


def test_prompt_for_the_iit_clause_mentions_the_deduction_subject() -> None:
    """The vocabulary must be able to express what 个税法第六条 is about."""
    messages = build_messages("中华人民共和国个人所得税法", [("c1", CLAUSE_IIT_6.section, CLAUSE_IIT_6.text)])
    assert "子女教育" in messages[0].content  # 税收事项 examples
    assert "赡养老人" in messages[0].content
    assert "专项附加扣除" in messages[1].content
