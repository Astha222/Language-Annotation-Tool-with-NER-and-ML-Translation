"""The GLiNER adapter: label mapping, output shape and the hook in eval_ner.make_pipeline (with a fake gliner module; nothing is downloaded)."""
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "backend"))


class FakeGliner:
    calls = []

    @classmethod
    def from_pretrained(cls, model_id):
        inst = cls()
        inst.model_id = model_id
        return inst

    def to(self, device):
        return self

    def predict_entities(self, text, labels, threshold=0.5):
        FakeGliner.calls.append(("single", text, tuple(labels), threshold))
        i = text.find("Rahul")
        return [{"start": i, "end": i + 5, "text": "Rahul", "label": "person", "score": 0.93},
                {"start": 0, "end": 1, "text": "x", "label": "date", "score": 0.99}] if i >= 0 else []


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setitem(sys.modules, "gliner", types.SimpleNamespace(GLiNER=FakeGliner))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    FakeGliner.calls.clear()
    import gliner_adapter
    return gliner_adapter


def test_pipeline_shape_label_mapping_and_unknown_labels_dropped(adapter):
    nlp = adapter.GlinerPipeline("some/gliner")
    out = nlp(["Rahul works here", "nobody"], batch_size=4)
    assert out == [[{"entity_group": "PER", "score": 0.93, "start": 0, "end": 5, "word": "Rahul"}], []]      # the "date" label is not scored


def test_it_asks_for_the_three_types_with_the_default_threshold(adapter):
    adapter.GlinerPipeline("some/gliner")(["Rahul"])
    assert FakeGliner.calls[0][2] == ("person", "organization", "location") and FakeGliner.calls[0][3] == 0.5


def test_threshold_can_be_set_by_environment_or_argument(adapter, monkeypatch):
    monkeypatch.setenv("GLINER_THRESHOLD", "0.3")
    assert adapter.GlinerPipeline("m").threshold == 0.3
    assert adapter.GlinerPipeline("m", threshold=0.7).threshold == 0.7


def test_a_single_string_returns_one_list_and_extra_pipeline_arguments_are_ignored(adapter):
    nlp = adapter.GlinerPipeline("m")
    assert nlp("Rahul", batch_size=1, aggregation_strategy="simple", stride=64)[0]["entity_group"] == "PER"


def test_missing_character_offsets_are_recovered_from_the_entity_text_or_dropped_and_counted(adapter):
    nlp = adapter.GlinerPipeline("m")
    text = "Rahul met Rahul and Mira"
    ents = [{"start": None, "end": None, "text": "Rahul", "label": "person", "score": 0.9},
            {"start": None, "end": None, "text": "Rahul", "label": "person", "score": 0.8},      # second one must take the second occurrence
            {"start": None, "end": None, "text": "Zed", "label": "person", "score": 0.7},         # not in the text: dropped
            {"start": 20, "end": 24, "text": "Mira", "label": "person", "score": 0.6}]
    out = nlp._convert(ents, text)
    assert [(e["start"], e["end"]) for e in out] == [(0, 5), (10, 15), (20, 24)]
    assert nlp.recovered == 2 and nlp.unresolved == 1


def test_run_all_routes_gliner_names_to_the_adapter(adapter):
    import eval_ner
    nlp, shown = eval_ner.make_pipeline(None, "bn", "gliner:knowledgator/gliner-x-base")
    assert isinstance(nlp, adapter.GlinerPipeline) and nlp.model.model_id == "knowledgator/gliner-x-base"
    assert shown == "gliner:knowledgator/gliner-x-base"


def test_locate_falls_back_to_case_folding_and_flexible_spaces(adapter):
    assert adapter.locate("Rahul met Rahul", "Rahul", {(0, 5)}) == (10, 15)          # first occurrence not taken
    assert adapter.locate("Er wohnt in BERLIN.", "berlin", set()) == (12, 18)        # case differs
    assert adapter.locate("New  York is big", "New York", set()) == (0, 9)           # two spaces in the text
    assert adapter.locate("nothing here", "Paris", set()) is None
    assert adapter.locate("text", "", set()) is None


def test_unplaced_entities_are_counted_with_examples(adapter):
    nlp = adapter.GlinerPipeline("m")
    nlp._convert([{"start": None, "end": None, "text": "Zed", "label": "person", "score": 0.7}], "Rahul ran")
    assert nlp.unresolved == 1 and nlp.unresolved_examples == [("Zed", "Rahul ran")]


class FakeOtter:
    revision = None

    @classmethod
    def from_pretrained(cls, model_id, trust_remote_code=False, revision=None):
        assert trust_remote_code
        FakeOtter.revision = revision
        return cls()

    def eval(self):
        return self

    def to(self, device):
        return self

    def predict(self, texts, labels, threshold=0.5, batch_size=16):
        return [[{"text": "Rahul", "label": "person", "start": 0, "end": 5, "score": 0.9}] if t.startswith("Rahul") else [] for t in texts]


def test_otter_is_routed_through_the_same_adapter_with_a_pinned_revision(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=FakeOtter))     # the fixture's fake torch stops the real one
    import eval_ner
    eval_ner._PIPELINES.pop("otter:whoisjones/otter-cross-mmbert", None)
    nlp, _ = eval_ner.make_pipeline(None, "de", "otter:whoisjones/otter-cross-mmbert")
    assert FakeOtter.revision == adapter.OTTER_REVISIONS["whoisjones/otter-cross-mmbert"]
    assert nlp(["Rahul geht", "nichts"]) == [[{"entity_group": "PER", "score": 0.9, "start": 0, "end": 5, "word": "Rahul"}], []]
