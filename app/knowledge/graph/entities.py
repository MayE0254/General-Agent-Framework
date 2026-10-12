"""Semantic entity layer for the fiscal/tax knowledge graph.

The structural layer (``app.knowledge.graph.extract``) records what the corpus
*literally says*: which clause cites which clause.  It cannot express the other
kind of relation -- two clauses that share a *subject* without naming each
other.  Measured on the corpus, 国发〔2022〕8号 and 国发〔2023〕13号 both cover
3 岁以下婴幼儿照护扣除 and neither cites the other; no amount of regular
expression recovers that link.

So this module calls an LLM per document and asks only one question: *which
entities of a fixed, closed type list does this clause mention?*  Everything
here except the call itself is a pure function over strings/dicts, which keeps
the prompt, the vocabulary and the parsing unit-testable without a network.

Three deliberate choices:

* **The type list is closed, the values are not.**  Entity *values* cannot be
  enumerated (非居民企业, 豆制品, ... are open sets), so what is closed is the
  set of five type keys.  A name is kept only if it maps onto one of them;
  anything else is dropped *and counted*, never written as a fallback type.
* **The wire format is an English key** (``taxpayer``/``tax_type``/...), with
  the Chinese gloss carried in the prompt.  Keys are a machine contract (Neo4j
  properties, validation, indexes), so they must not drift with hyphenation or
  full/half-width punctuation; the model is still free to *answer* with the
  Chinese label, which is normalised back to the key.
* **No relation between entities is extracted here.**  The only edge this layer
  produces is clause->entity.  Clause->clause is then a two-hop traversal in
  the retrieval direction, which keeps the graph free of the O(n^2) junk edges
  a modelled ``RELATED_TO`` would produce.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from app.llm.schemas import LLMMessage, LLMRole

# --------------------------------------------------------------------------
# Closed vocabulary
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EntityTypeSpec:
    """One entry of the closed type list: machine key + how to prompt for it."""

    key: str
    label: str
    gloss: str
    examples: str


ENTITY_TYPES: tuple[EntityTypeSpec, ...] = (
    EntityTypeSpec(
        key="taxpayer",
        label="纳税人主体",
        gloss="承担纳税义务或扣缴义务、或者享受某项税收待遇的主体",
        examples="非居民企业、居民企业、个体工商户、扣缴义务人、小型微利企业、高新技术企业",
    ),
    EntityTypeSpec(
        key="tax_type",
        label="税种",
        gloss="税收法律、行政法规所设立的税种",
        examples="企业所得税、个人所得税、增值税、消费税、环境保护税、印花税",
    ),
    EntityTypeSpec(
        key="tax_matter",
        label="税收事项",
        gloss="针对某个税种的优惠、扣除、减免、税率待遇或计税规则",
        examples="专项附加扣除、子女教育、赡养老人、3岁以下婴幼儿照护、减按10%的税率、加计扣除、免征",
    ),
    EntityTypeSpec(
        key="tax_base",
        label="征税对象",
        gloss="计税依据或征税范围所指向的货物、所得、财产或行为",
        examples="豆制品、精制茶、农产品、应税大气污染物、工资薪金所得",
    ),
    EntityTypeSpec(
        key="admin",
        label="征管事项",
        gloss="程序性事项",
        examples="纳税申报、备案、发票、退税、汇算清缴、税收保全",
    ),
)

ENTITY_TYPE_KEYS: tuple[str, ...] = tuple(spec.key for spec in ENTITY_TYPES)
# The model is asked for the machine key, but a Chinese label is accepted on
# the way back: rejecting it would turn a formatting slip into a lost entity.
_TYPE_ALIASES: dict[str, str] = {}
for _spec in ENTITY_TYPES:
    _TYPE_ALIASES[_spec.key] = _spec.key
    _TYPE_ALIASES[_spec.label] = _spec.key
del _spec

# Relationship type written for every clause->entity mention.  A constant, like
# the structural layer's labels: it is part of the graph contract, not config.
MENTIONS = "MENTIONS"

MAX_ENTITIES_PER_CLAUSE = 5
MIN_NAME_CHARS = 2
MAX_NAME_CHARS = 20


def resolve_type_key(raw: str) -> str | None:
    """Map a model-supplied type token onto the closed vocabulary.

    Returns ``None`` for anything outside the list, so the caller can drop and
    count it.  There is deliberately no catch-all type: a miscellaneous bucket
    is where junk accumulates, and it would make the vocabulary unfalsifiable.
    """
    token = re.sub(r"\s+", "", raw or "").strip("`\"'。.：:")
    if not token:
        return None
    return _TYPE_ALIASES.get(token) or _TYPE_ALIASES.get(token.lower())


_STRIP_CHARS = "《》〈〉「」『』【】()（）[]〔〕“”\"'、,，。.；;：:!！?？…·-—_/\\|"
_TRAILING_NOISE = ("等", "的", "相关", "有关规定")


def normalize_entity_name(raw: str) -> str:
    """Canonicalise a surface form so equal entities merge into one node.

    Only mechanical normalisation: whitespace, wrapping punctuation and
    book-title marks.  It never rewrites the wording -- 非居民企业 and 居民企业
    look 2 characters apart and mean opposite things, so synonym folding is a
    decision the extraction prompt has to make, not a string edit here.
    """
    name = re.sub(r"\s+", "", raw or "").strip()
    name = name.strip(_STRIP_CHARS)
    for noise in _TRAILING_NOISE:
        if len(name) > len(noise) + MIN_NAME_CHARS and name.endswith(noise):
            name = name[: -len(noise)]
    return name.strip(_STRIP_CHARS)


def entity_id(type_key: str, name: str) -> str:
    """Stable node id: the type is part of the key.

    个人所得税 as a 税种 and as a 税收事项 (the name of the law's subject) are
    different nodes; collapsing them by name alone would mix two vocabularies
    into one node and make the ``type`` property meaningless.
    """
    return f"{type_key}:{name}"


# --------------------------------------------------------------------------
# Extraction shapes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EntityMention:
    """One ``clause mentions entity`` fact.

    ``extractor`` is the model id that produced the mention, stamped by the
    caller.  It rides into the graph so a run that degraded to the fallback leg
    can still be audited edge by edge.
    """

    clause_id: str
    type_key: str
    name: str
    extractor: str = ""

    @property
    def entity_id(self) -> str:
        return entity_id(self.type_key, self.name)


@dataclass
class EntityExtraction:
    """The entity layer of one document, plus how much had to be discarded."""

    doc_id: str
    mentions: list[EntityMention] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def entities(self) -> dict[str, EntityMention]:
        """Distinct entities keyed by node id (last mention wins)."""
        return {m.entity_id: m for m in self.mentions}


class EntityParseError(ValueError):
    """The model's answer was not usable JSON.

    Raised rather than returning an empty extraction: an empty extraction looks
    exactly like "this document has no entities", which would silently delete
    the document's entity layer from the graph.
    """


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------
_SYSTEM_PROMPT = """你是财税法规知识图谱的实体抽取器。任务：从给定的法规条款正文中，识别出属于下列**封闭类型表**的实体，逐条款输出 JSON。

## 实体类型表（只能使用这 5 个 key，不得新增类型，不得自造类型）
{type_table}

## 抽取规则
1. 只抽该条款正文中**字面出现**的实体，只做名称识别，不做推理、不补充外部背景知识。
2. 不要抽取：法规名称、发文字号（如"国发〔2023〕13号"）、发文机关、日期、金额、数量、百分比数字；也不要输出条款之间的引用关系。
3. 名称照抄正文用词，去掉动词、修饰语与标点，长度 2~20 字，使用规范全称（例如统一写"个人所得税"，不要写"个税"）。
4. 同一条款内同一名称只输出一次；名称必须能归入上表某一类型，归不进去的不要输出。
5. 一条条款最多输出 {max_per_clause} 个实体；确实没有实体就返回空数组，**不要为了凑数硬填**。
6. 我给出的每一个条款编号都必须在输出中出现；没有实体的条款用空数组占位。

## 输出格式（只输出 JSON 对象，不要解释、不要 Markdown 代码块）
{{"clauses":[{{"ref":"c1","entities":[{{"type":"taxpayer","name":"非居民企业"}}]}}]}}"""


def _render_type_table() -> str:
    lines = []
    for spec in ENTITY_TYPES:
        lines.append(f"- {spec.key} {spec.label}：{spec.gloss}。例：{spec.examples}")
    return "\n".join(lines)


def build_messages(
    title: str, clauses: Iterable[tuple[str, str, str]]
) -> list[LLMMessage]:
    """Render the extraction request for one batch of clauses.

    ``clauses`` is a sequence of ``(ref, section, text)``.  ``ref`` is a short
    local id (``c1``/``c2``/...) rather than a clause id: the model has to echo
    it back verbatim, and a 20-character hash is something models mangle.  The
    ref -> clause_id map stays on this side, so a mangled ref costs one clause
    instead of corrupting a node id.
    """
    system = _SYSTEM_PROMPT.format(
        type_table=_render_type_table(),
        max_per_clause=MAX_ENTITIES_PER_CLAUSE,
    )
    blocks = []
    for ref, section, text in clauses:
        head = f"【{ref}】" + (f"{section} " if section else "")
        blocks.append(head + (text or "").strip())
    user = (
        f"文档标题：{title}\n\n以下每段以【条款编号】开头，逐条抽取实体：\n\n"
        + "\n\n".join(blocks)
    )
    return [
        LLMMessage(role=LLMRole.SYSTEM, content=system),
        LLMMessage(role=LLMRole.USER, content=user),
    ]


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------
_FENCE_RX = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TRAILING_COMMA_RX = re.compile(r",\s*([}\]])")


def _loads_object(content: str) -> dict:
    """Parse a JSON object out of a model answer, tolerating prose and fences.

    Three attempts, cheapest first: the raw answer, a fenced block inside it,
    then the outermost ``{...}`` span with trailing commas removed.  Anything
    still unparsable is a hard error -- see ``EntityParseError``.
    """
    text = (content or "").strip()
    if not text:
        raise EntityParseError("empty response")
    candidates = [text]
    fenced = _FENCE_RX.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
        candidates.append(_TRAILING_COMMA_RX.sub(r"\1", text[start : end + 1]))
    last_error = "no JSON object found"
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if isinstance(parsed, dict):
            return parsed
        last_error = f"top-level {type(parsed).__name__}, expected object"
    raise EntityParseError(last_error)


def parse_extraction(
    content: str, ref_to_clause: dict[str, str], doc_id: str
) -> EntityExtraction:
    """Validate a model answer against the vocabulary and the ref map.

    Every rejection is counted by reason.  The counts matter more than they
    look: a model that starts answering with an invented type would otherwise
    degrade the graph quietly, with the only symptom being fewer edges than
    expected several hours later.
    """
    payload = _loads_object(content)
    raw_clauses = payload.get("clauses")
    if not isinstance(raw_clauses, list):
        raise EntityParseError("missing 'clauses' array")

    stats = {
        "refs_sent": len(ref_to_clause),
        "refs_returned": 0,
        "refs_missing": 0,
        "refs_unknown": 0,
        "raw_entities": 0,
        "unknown_types": 0,
        "invalid_names": 0,
        "duplicates": 0,
        "over_limit": 0,
    }
    mentions: list[EntityMention] = []
    seen_refs: set[str] = set()

    for item in raw_clauses:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        clause_id = ref_to_clause.get(ref)
        if ref in seen_refs:
            continue
        seen_refs.add(ref)
        if clause_id is None:
            stats["refs_unknown"] += 1
            continue
        stats["refs_returned"] += 1
        entities = item.get("entities")
        if not isinstance(entities, list):
            continue
        # Deduplication is per clause, never across the batch: the same entity
        # reached from two clauses is the one fact that bridges them, and a
        # batch-wide ``seen`` set would drop the second mention and silently
        # delete every cross-clause path this layer exists to build.
        seen_pairs: set[tuple[str, str]] = set()
        kept = 0
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            stats["raw_entities"] += 1
            type_key = resolve_type_key(str(entity.get("type") or ""))
            if type_key is None:
                stats["unknown_types"] += 1
                continue
            name = normalize_entity_name(str(entity.get("name") or ""))
            if not (MIN_NAME_CHARS <= len(name) <= MAX_NAME_CHARS):
                stats["invalid_names"] += 1
                continue
            pair = (type_key, name)
            if pair in seen_pairs or kept >= MAX_ENTITIES_PER_CLAUSE:
                stats["duplicates" if pair in seen_pairs else "over_limit"] += 1
                continue
            seen_pairs.add(pair)
            kept += 1
            mentions.append(
                EntityMention(clause_id=clause_id, type_key=type_key, name=name)
            )

    stats["refs_missing"] = len(ref_to_clause) - stats["refs_returned"]
    stats["mentions"] = len(mentions)
    stats["entities"] = len({m.entity_id for m in mentions})
    return EntityExtraction(doc_id=doc_id, mentions=mentions, stats=stats)


def is_truncated(finish_reason: str, content: str) -> bool:
    """Whether the answer was cut off, so the batch must be re-sent smaller.

    ``finish_reason == "length"`` is the provider saying the ``max_tokens`` cap
    bit.  A reasoning model can also stop "normally" on the answer channel while
    the JSON is still open, so an unclosed object counts too.
    """
    if finish_reason == "length":
        return True
    text = (content or "").strip()
    if not text:
        return False
    return text.count("{") > text.count("}")


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClauseInput:
    """One clause as the extractor sees it (a projection of ClauseRecord)."""

    clause_id: str
    doc_id: str
    title: str
    section: str
    text: str


def split_batch(
    clauses: list[ClauseInput], *, max_clauses: int, max_chars: int
) -> list[list[ClauseInput]]:
    """Split one document's clauses into call-sized batches.

    Both caps exist for different reasons: the clause count bounds the answer
    length (the model must echo every ref), the character budget bounds the
    prompt.  13 of the 4189 corpus documents exceed 24k characters, one of them
    a 1212-clause index, so without this the run would fail on a handful of
    files and look "mostly fine".
    """
    batches: list[list[ClauseInput]] = []
    current: list[ClauseInput] = []
    chars = 0
    for clause in clauses:
        size = len(clause.text) + len(clause.section) + 8
        if current and (len(current) >= max_clauses or chars + size > max_chars):
            batches.append(current)
            current, chars = [], 0
        current.append(clause)
        chars += size
    if current:
        batches.append(current)
    return batches


def assign_refs(clauses: list[ClauseInput]) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Number a batch's clauses and return ``(ref -> clause_id, prompt rows)``."""
    ref_to_clause: dict[str, str] = {}
    rows: list[tuple[str, str, str]] = []
    for index, clause in enumerate(clauses, start=1):
        ref = f"c{index}"
        ref_to_clause[ref] = clause.clause_id
        rows.append((ref, clause.section, clause.text))
    return ref_to_clause, rows
