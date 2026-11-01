import json
import pytest

from dashboard import collect as vo
from dashboard import ratings as rt
from dashboard.server import State


def match(a, b, a_wins, b_wins):
    return {"a": a, "b": b, "a_wins": a_wins, "b_wins": b_wins, "draws": 0}


def test_even_record_fits_to_zero_and_a_lead_to_positive():
    assert rt.fit([match("x", "anchor", 10, 10)], "anchor")["x"] == pytest.approx(0, abs=1e-6)
    assert rt.fit([match("x", "anchor", 30, 10)], "anchor")["x"] > 0


def test_more_games_shrink_the_standard_error():
    def error(scale):
        games = [match("x", "anchor", 3 * scale, 2 * scale)]
        return rt.standard_errors(games, rt.fit(games, "anchor"), "anchor")["x"]

    assert error(10) < error(1)
    assert error(40) == pytest.approx(error(10) / 2, rel=0.1)


def test_undefeated_model_is_only_a_bound():
    games = [match("x", "anchor", 48, 0), match("y", "anchor", 30, 18)]
    assert rt.only_a_bound(games, "x")
    assert not rt.only_a_bound(games, "y")


def test_disconnected_models_are_flagged():
    games = [match("x", "anchor", 5, 5), match("p", "q", 5, 5)]
    table = {r["model"]: r for r in vo.rating_table([{**games[0], "a": "x", "b": vo.ANCHOR}, games[1]])}
    assert table["x"]["connected"]
    assert not table["p"]["connected"]


def test_wilson_interval_contains_the_score():
    low, high = rt.wilson(51, 80)
    assert low < 51 / 80 < high
    assert rt.wilson(0, 0) == (0.0, 1.0)


def test_records_reads_pretty_printed_and_line_delimited(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{\n  "b": {"c": 2}\n}\nnot json\n{"d": 3}')
    assert vo.records(path) == [{"a": 1}, {"b": {"c": 2}}, {"d": 3}]


def arena(tmp_path, rel, wins, losses, **extra):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    games = wins + losses
    path.write_text(json.dumps({
        "schema": "vgo.arena.v1", "games": games, "candidate_wins": wins,
        "candidate_losses": losses, "candidate_score": wins / games,
        "simulations_per_move": 800, **extra,
    }))


def test_ledger_shows_the_median_repeat_rather_than_pooling(tmp_path):
    for i, wins in enumerate([35, 37, 39], 1):
        arena(tmp_path, f"artifacts/depth-ab/h2h-{i}.json", wins, 60 - wins)
    rows = vo.ledger(vo.arena_records(tmp_path))
    assert len(rows) == 1
    assert rows[0]["record"] == "37-23"
    assert rows[0]["runs"] == 3


def test_ledger_verdicts_follow_the_interval(tmp_path):
    arena(tmp_path, "artifacts/a/one.json", 70, 10)
    arena(tmp_path, "artifacts/b/two.json", 10, 70)
    arena(tmp_path, "artifacts/c/three.json", 40, 40)
    verdicts = {r["experiment"]: r["verdict"] for r in vo.ledger(vo.arena_records(tmp_path))}
    assert verdicts == {"a": "Candidate better", "b": "Candidate worse", "c": "Inconclusive"}


def test_small_arenas_stay_out_of_the_ledger(tmp_path):
    arena(tmp_path, "artifacts/a/one.json", 4, 2)
    assert vo.ledger(vo.arena_records(tmp_path)) == []


def test_sample_rate_skips_gaps_between_updates():
    updates = [
        {"update": 1, "cumulative": 1000, "finished": 0},
        {"update": 2, "cumulative": 3000, "finished": 3600},
        {"update": 4, "cumulative": 9000, "finished": 7200},
    ]
    assert vo.samples_per_hour(updates) == [{"update": 2, "rate": 2000.0}]


def test_collect_tolerates_an_empty_project(tmp_path):
    result = vo.collect(tmp_path)
    assert [s["id"] for s in result["sections"]] == ["speed", "signal", "model", "spec", "next"]


@pytest.fixture
def state(tmp_path):
    return State(tmp_path, tmp_path / "ideas.json")


def test_ideas_can_be_added_edited_and_deleted(state):
    idea = state.save_idea({"fields": {"title": "First", "bucket": "Speed"}})[0]
    assert idea["status"] == "Next"
    edited = state.save_idea({"id": idea["id"], "fields": {"status": "Done", "result": "Won"}})[0]
    assert (edited["title"], edited["status"], edited["result"]) == ("First", "Done", "Won")
    assert state.save_idea({"id": idea["id"], "delete": True}) == []


def test_unknown_idea_fields_are_dropped(state):
    idea = state.save_idea({"fields": {"title": "x", "evil": "y"}})[0]
    assert "evil" not in idea


def test_metrics_are_cached_between_requests(state, monkeypatch):
    calls = []
    monkeypatch.setattr("dashboard.server.collect", lambda p: calls.append(p) or {})
    state.data()
    state.data()
    assert len(calls) == 1


def test_lemma_status_reads_the_spec(tmp_path):
    (tmp_path / "spec").mkdir()
    (tmp_path / "spec/LEMMAS.md").write_text(
        "## A1. Plain\n\nproof\n\n### E6. Snap\n\n*Status: Conditional on stones.* Used.\n\n"
        "### O1. Gap\n\n*Status: Open.* text\n")
    assert [(r["id"], r["status"]) for r in vo.lemma_status(tmp_path)] == [
        ("A1", "Proved"), ("E6", "Conditional"), ("O1", "Open")]
