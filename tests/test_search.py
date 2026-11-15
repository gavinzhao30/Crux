import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from core.config import RunConfig
from core.schemas import Experiment, OutcomeBin
from search.exa import ExaSearch, SearchUnavailable
from search.experiments import SearchExperiments, evaluate_rule
from search.policy import PolicyDenied, SearchPolicy

QUERY = "DESeq2 documentation alpha default"
DOC_URL = "https://bioconductor.org/packages/release/bioc/manuals/DESeq2/man/DESeq2.pdf"


def policy(**kwargs):
    return SearchPolicy(task_id="synthetic-demo", forbidden_terms=("private-study-zeta",), **kwargs)


def hit(url=DOC_URL, text="results(object, alpha = 0.1)", title="DESeq2 reference"):
    return SimpleNamespace(url=url, title=title, id=url, highlights=[text])


class FakeSDK:
    def __init__(self, results=None, cost=0.007):
        self.results = [hit()] if results is None else results
        self.calls = []
        self.cost = cost
        self.failure = None
        self.before_call = None

    async def search(self, **request):
        self.calls.append(request)
        if self.before_call:
            self.before_call()
        if self.failure:
            raise self.failure
        return SimpleNamespace(
            results=self.results,
            cost_dollars=SimpleNamespace(total=self.cost),
            request_id="req-demo",
        )


def experiment():
    return Experiment(
        id="search-1",
        kind="search",
        question_ids=["q-alpha"],
        goal="Check tool default",
        query=QUERY,
        bins=[
            OutcomeBin(
                id="alpha-01",
                description="Only alpha 0.1 appears",
                rule="'alpha = 0.1' in text and 'alpha = 0.05' not in text",
            ),
            OutcomeBin(
                id="alpha-005",
                description="Only alpha 0.05 appears",
                rule="'alpha = 0.05' in text and 'alpha = 0.1' not in text",
            ),
        ],
    )


@pytest.mark.parametrize(
    "query",
    [
        "DESeq2 docs private-study-zeta results",
        "DESeq2 defaults 10.1234/private-paper",
        "DESeq2 manual GSE12345",
        "DESeq2 defaults %47%53%45%31%32%33",
        "DESeq2 docs G\u200bSE123",
        "DESeq2 docs source paper",
        "DESeq2 docs preprint",
        "DESeq2 docs ground_truth",
        "DESeq2 documentation answer key",
        "DESeq2 docs site:evil.example",
        "DESeq2 docs https://evil.example",
        "What are the biological findings?",
        "",
        "x" * 2001,
    ],
)
def test_queries_are_denied(query):
    with pytest.raises(PolicyDenied):
        policy().check_query(query)


@pytest.mark.parametrize(
    "url",
    [
        "https://bioconductor.org.evil.example/packages/DESeq2",
        "https://evil.example/bioconductor.org/packages/DESeq2",
        "https://bioconductor.org@evil.example/packages/DESeq2",
        "http://bioconductor.org/packages/DESeq2",
        "https://bioconductor.org:444/packages/DESeq2",
        "https://bioconductor.org/packages/../paper",
        "https://bioconductor.org/packages/%2e%2e/paper",
        "https://bioconductor.org/packages/%255cprivate",
        "https://bioconductor.org/packages-old/paper",
        "https://bioconductor.org/packages/DESeq2?url=https://evil.example",
        "https://bioconductor.org/packages/private-study-zeta",
        "https://bioconductor.org/packages/DESeq2/supplement",
        "https://github.com/unapproved-owner/study-results",
        "https://storage.googleapis.com/bixbench3-ground-truth/task9",
        "https://doi.org/10.1234/study",
        "https://arxiv.org/abs/1234.5678",
    ],
)
def test_urls_are_denied(url):
    with pytest.raises(PolicyDenied):
        policy().check_url(url)


def test_allowed_queries_paths_and_required_task_exclusions():
    assert policy().check_query("  DESeq2   documentation alpha default ") == QUERY
    policy().check_url(DOC_URL + "#alpha")
    policy().check_url("https://github.com/bioc/DESeq2/wiki")
    with pytest.raises(ValueError):
        SearchPolicy(task_id="task9", forbidden_terms=())
    with pytest.raises(ValueError):
        SearchPolicy(task_id="task9", forbidden_terms=(" ",))


def test_sdk_filters_exclude_task_sources():
    configured = policy(forbidden_sources=("bioconductor.org/packages/private-paper",))
    assert "bioconductor.org/packages" in configured.sdk_filters()["include_domains"]
    assert "bioconductor.org/packages/private-paper" in configured.sdk_filters()["exclude_domains"]
    with pytest.raises(PolicyDenied):
        configured.check_url("https://bioconductor.org/packages/private-paper/figures")


def test_cache_survives_restart_and_charges_only_first_lookup(tmp_path):
    async def scenario():
        sdk = FakeSDK()
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
            first = await search.lookup(QUERY)
            assert first.cost_usd == 0.007 and not first.cache_hit
            assert first.evidence[0].kind == "search"
            assert first.evidence[0].author_role == "experimenter"
            assert sdk.calls[0]["contents"] == {"highlights": True}
            assert sdk.calls[0]["type"] == "auto"
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as restored:
            second = await restored.lookup(QUERY)
            assert second.cost_usd == 0 and second.cache_hit
            assert second.evidence == first.evidence
            assert len(sdk.calls) == 1
            events = [
                json.loads(row[0])
                for row in restored.db.execute("SELECT payload FROM audit WHERE type='search.done'")
            ]
            assert [e["cost_usd"] for e in events] == [0.007, 0.0]

    asyncio.run(scenario())


def test_policy_change_cannot_reuse_old_cached_evidence(tmp_path):
    async def scenario():
        sdk = FakeSDK()
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
            assert await search.find(QUERY)
        strict = SearchPolicy(task_id="synthetic-demo", forbidden_terms=("DESeq2",))
        async with ExaSearch(strict, "r1", tmp_path, client=sdk) as search:
            with pytest.raises(PolicyDenied):
                await search.find(QUERY)
        assert len(sdk.calls) == 1

    asyncio.run(scenario())


def test_blocked_results_never_enter_cache_or_evidence(tmp_path):
    async def scenario():
        sdk = FakeSDK(
            [
                hit(),
                hit("https://evil.example/paper", "LEAKED ANSWER"),
                hit(text="private-study-zeta LEAKED SECRET"),
                hit(text=""),
                hit(),
            ]
        )
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
            result = await search.lookup(QUERY)
            assert len(result.evidence) == 1
            cached = search.db.execute("SELECT result FROM cache").fetchone()[0]
            assert "LEAKED" not in cached
            audits = list(
                search.db.execute("SELECT payload FROM audit WHERE type='search.result_blocked'")
            )
            assert len(audits) == 3
            assert any("evil.example" in r[0] for r in audits)
            assert not any("LEAKED" in r[0] for r in audits)

    asyncio.run(scenario())


def test_disabled_search_needs_no_key_and_never_calls_sdk(tmp_path, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    async def scenario():
        sdk = FakeSDK()
        async with ExaSearch(
            policy(), "r1", tmp_path, config=RunConfig(search=False), client=sdk
        ) as search:
            assert await search.find(QUERY) == []
            assert not (await SearchExperiments(search).run(experiment())).valid
            assert not sdk.calls

    asyncio.run(scenario())


def test_missing_key_is_explicit_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    async def scenario():
        async with ExaSearch(policy(), "r1", tmp_path) as search:
            with pytest.raises(SearchUnavailable, match="EXA_API_KEY"):
                await search.find(QUERY)
            assert search.db.execute("SELECT count(*) FROM cache").fetchone()[0] == 0

    asyncio.run(scenario())


def test_failures_are_not_empty_matches_and_cost_is_unknown(tmp_path):
    async def scenario():
        sdk = FakeSDK()
        sdk.failure = TimeoutError("offline")
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
            with pytest.raises(SearchUnavailable):
                await search.find(QUERY)
            error = json.loads(
                search.db.execute("SELECT payload FROM audit WHERE type='search.error'").fetchone()[
                    0
                ]
            )
            assert error["cost_usd"] is None
            sdk.failure = None
            sdk.results = []
            assert await search.find(QUERY) == []

    asyncio.run(scenario())


def test_concurrent_identical_queries_share_one_request(tmp_path):
    async def scenario():
        sdk = FakeSDK()
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
            results = await asyncio.gather(*(search.lookup(QUERY) for _ in range(4)))
            assert len(sdk.calls) == 1
            assert sum(r.cost_usd for r in results) == 0.007

    asyncio.run(scenario())


def test_missing_provider_price_is_labelled_estimate(tmp_path):
    async def scenario():
        async with ExaSearch(
            policy(),
            "r1",
            tmp_path,
            client=FakeSDK(cost=None),
            estimated_cost_usd=0.009,
        ) as search:
            result = await search.lookup(QUERY)
            assert result.cost_usd == 0.009
            assert result.cost_basis == "configured_estimate"

    asyncio.run(scenario())


def test_spec_is_committed_before_search_and_outcome_is_computed_by_code(tmp_path):
    async def scenario():
        sdk = FakeSDK()
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:

            def before_call():
                row = search.db.execute("SELECT * FROM experiments").fetchone()
                assert row["spec_hash"] and row["spec"] and row["observation"] is None

            sdk.before_call = before_call
            runner = SearchExperiments(search)
            obs = await runner.run(experiment())
            assert obs.valid and obs.bin_id == "alpha-01" and obs.cost_usd == 0.007
            runner.record_belief_effect("search-1", "q-alpha", 0.5)
            with pytest.raises(ValueError, match="spent"):
                await runner.run(experiment())
            changed_id = experiment().model_copy(update={"id": "same-spec-new-id"})
            with pytest.raises(ValueError, match="spent"):
                await runner.run(changed_id)
            assert len(sdk.calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "results",
    [[], [hit(text="alpha = 0.1 and alpha = 0.05")], [hit(text="No default is documented")]],
)
def test_missing_or_conflicting_evidence_cannot_update_beliefs(tmp_path, results):
    async def scenario():
        async with ExaSearch(policy(), "r1", tmp_path, client=FakeSDK(results)) as search:
            runner = SearchExperiments(search)
            obs = await runner.run(experiment())
            assert not obs.valid and obs.bin_id is None
            with pytest.raises(ValueError, match="invalid"):
                runner.record_belief_effect("search-1", "q-alpha", 0.5)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "rule",
    [
        "__import__('os').system('whoami')",
        "text.__class__",
        "text[0]",
        "[x for x in text]",
        "len(text) > 0",
        "1 + 1 == 2",
        "text",
        "n_sources > 'invalid'",
        "broken syntax(",
    ],
)
def test_rule_language_rejects_executable_python_and_bad_types(rule):
    with pytest.raises(ValueError):
        evaluate_rule(rule, {"text": "docs", "n_sources": 1})


def test_real_async_sdk_contract_with_mock_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")

    async def scenario():
        from exa_py import AsyncExa

        def handle(request):
            assert request.url == "https://api.exa.ai/search"
            body = json.loads(request.content)
            assert body["includeDomains"] == list(policy().allowed_sources)
            assert body["contents"] == {"highlights": True}
            assert request.headers["x-api-key"] == "fake-exa-key"
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "doc1",
                            "title": "DESeq2 docs",
                            "url": DOC_URL,
                            "highlights": ["alpha = 0.1"],
                        }
                    ],
                    "costDollars": {"total": 0.007},
                },
            )

        async with AsyncExa(timeout=2) as sdk:
            sdk._client = httpx.AsyncClient(
                base_url="https://api.exa.ai",
                headers=sdk.headers,
                transport=httpx.MockTransport(handle),
            )
            async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
                result = await search.lookup(QUERY)
                assert result.evidence[0].claim == "alpha = 0.1"
                assert result.cost_usd == 0.007

    asyncio.run(scenario())


def test_failed_remote_experiment_reserves_cost_without_a_belief_update(tmp_path):
    async def scenario():
        sdk = FakeSDK()
        sdk.failure = TimeoutError("provider may still charge")
        async with ExaSearch(policy(), "r1", tmp_path, client=sdk) as search:
            observation = await SearchExperiments(search).run(experiment())
            assert not observation.valid and observation.bin_id is None
            assert observation.cost_usd == 0.007
            assert observation.value["actual_cost_known"] is False

    asyncio.run(scenario())
