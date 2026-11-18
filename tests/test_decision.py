import pytest

from traffic_ai.decision import (
    Action,
    DecisionConfig,
    DecisionEngine,
    Detection,
    parse_speed_limit,
)

FRAME = (640, 640)
BOX = (100.0, 100.0, 200.0, 200.0)  # 100x100 px, well above the size filter


def det(label: str, conf: float = 0.9, box=BOX) -> Detection:
    return Detection(label, conf, box)


def run(engine: DecisionEngine, frames: list[list[Detection]]):
    decision = None
    for dets in frames:
        decision = engine.update(dets, FRAME)
    return decision


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine(DecisionConfig(window=5, min_hits=3, stop_hold_frames=4))


def test_parse_speed_limit():
    assert parse_speed_limit("Speed Limit 50") == 50
    assert parse_speed_limit("Speed Limit 120") == 120
    assert parse_speed_limit("Stop") is None


def test_single_frame_false_positive_is_ignored(engine):
    decision = run(engine, [[det("Red Light")], [], [], []])
    assert decision.action is Action.CRUISE


def test_red_light_confirmed_after_min_hits(engine):
    decision = run(engine, [[det("Red Light")]] * 3)
    assert decision.action is Action.STOP
    assert decision.target_speed == 0


def test_flickering_detection_still_confirms(engine):
    # Seen in 3 of 5 frames, with gaps: still confirmed.
    frames = [[det("Red Light")], [], [det("Red Light")], [], [det("Red Light")]]
    assert run(engine, frames).action is Action.STOP


def test_red_beats_green(engine):
    both = [det("Red Light"), det("Green Light")]
    decision = run(engine, [both] * 3)
    assert decision.action is Action.STOP
    assert "conflicting" in decision.reason


def test_green_light_proceeds(engine):
    assert run(engine, [[det("Green Light")]] * 3).action is Action.PROCEED


def test_low_confidence_green_is_rejected_but_red_is_accepted(engine):
    assert run(engine, [[det("Green Light", conf=0.4)]] * 3).action is Action.CRUISE
    engine.reset()
    assert run(engine, [[det("Red Light", conf=0.4)]] * 3).action is Action.STOP


def test_tiny_far_away_detection_is_ignored(engine):
    tiny = (0.0, 0.0, 5.0, 5.0)  # 25 px^2 on a 640x640 frame
    assert run(engine, [[det("Red Light", box=tiny)]] * 3).action is Action.CRUISE


def test_speed_limit_persists_after_sign_disappears(engine):
    run(engine, [[det("Speed Limit 50")]] * 3)
    decision = run(engine, [[]] * 10)
    assert decision.speed_limit == 50
    assert decision.target_speed == 50
    assert decision.action is Action.CRUISE


def test_speed_limit_is_updated_by_new_sign(engine):
    run(engine, [[det("Speed Limit 50")]] * 3)
    run(engine, [[]] * 5)
    decision = run(engine, [[det("Speed Limit 90")]] * 3)
    assert decision.speed_limit == 90


def test_stop_sign_holds_then_releases(engine):
    stop = [[det("Stop")]] * 3
    assert run(engine, stop).action is Action.STOP
    # Sign leaves the view: the hold (4 frames total) keeps the vehicle stopped for a while...
    actions = [engine.update([], FRAME).action for _ in range(6)]
    assert actions[0] is Action.STOP
    # ...and then it is released.
    assert actions[-1] is Action.CRUISE


def test_stop_sign_does_not_retrigger_while_still_visible(engine):
    # Sign stays in view for a long time: stop once, then continue.
    decisions = [engine.update([det("Stop")], FRAME) for _ in range(20)]
    assert decisions[2].action is Action.STOP
    assert decisions[-1].action is Action.CRUISE


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        DecisionConfig(window=3, min_hits=5)
