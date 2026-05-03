"""
Unit tests for extract_choice — the MCQ-answer parser used by the benchmark
runner. Covers the 3 fixes applied after the deepseek thinking benchmark:

  1. Markdown decoration (** *, _, ` , #) stripped before pattern matching
     so "**Answer:** B" works.
  2. Pattern 6 (verbatim choice) uses word boundaries so "Class A" doesn't
     false-positive inside "class addresses".
  3. Refusal phrases ("Context does not cover", "None of the options",
     "Cannot determine", "Insufficient context") short-circuit before the
     fuzzy fallback, preventing hallucinated choice picks.
"""
import sys
from pathlib import Path

# Make the benchmark module importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "benchmark"))

from run_teleqna import extract_choice, _is_refusal  # noqa: E402


# ---------- Markdown stripping (Fix 2) -------------------------------------


class TestMarkdownStripping:
    def test_double_asterisk_around_answer(self):
        text = "**Answer:** B\n\n**Justification:** ..."
        choices = ["alpha foo", "bravo bar", "charlie baz", "delta qux"]
        assert extract_choice(text, choices) == 1

    def test_heading_prefix(self):
        text = "### Answer: C\n\nSome context."
        choices = ["a", "b", "c", "d"]
        # "a"/"b"/"c"/"d" are too short for pattern 6 (<6 chars) but pattern 2 fires.
        assert extract_choice(text, choices) == 2

    def test_inline_emphasis(self):
        text = "**Answer: C**\n\nThe correct option is C."
        choices = ["alpha", "bravo", "charlie", "delta"]
        assert extract_choice(text, choices) == 2

    def test_blockquote_prefix(self):
        text = "> Answer: D\nJustification: ..."
        choices = ["a one", "b two", "c three", "d four"]
        assert extract_choice(text, choices) == 3


# ---------- Pattern 6 word boundary (Fix 1) --------------------------------


class TestPattern6WordBoundary:
    def test_no_false_positive_for_compound_word(self):
        # Real benchmark case: choice "Class A" was matched inside
        # "class addresses" → false positive pred=4, masquerading as correct.
        text = (
            "The context provided does not cover information related to TCP/IP "
            "class addresses or network addressing schemes. Therefore, it's not "
            "possible to answer based on the given context.\n\n"
            "Answer: Context does not cover which class has the largest number "
            "of possible network addresses in TCP/IP."
        )
        choices = ["Class E", "Class B", "Class C", "Class D", "Class A"]
        # Refusal short-circuits at step 2.5, returns None.
        assert extract_choice(text, choices) is None

    def test_word_boundary_still_matches_real_phrase(self):
        # When "Class A" appears as a real phrase (not embedded in a longer
        # word), pattern 6 should still pick it.
        text = "Looking at the options, Class A network is the answer."
        choices = ["Class E network", "Class B network", "Class C network", "Class D network", "Class A network"]
        assert extract_choice(text, choices) == 4

    def test_pattern_6_prefers_first_occurrence(self):
        # "alpha service" appears before "bravo service" → first wins.
        text = "Looking at alpha service requirements then bravo service later."
        choices = ["alpha service", "bravo service", "charlie service"]
        assert extract_choice(text, choices) == 0


# ---------- Refusal short-circuit (Fix 3) ----------------------------------


class TestRefusalShortCircuit:
    @staticmethod
    def _generic_choices():
        return [
            "alpha service definition",
            "bravo service definition",
            "charlie service definition",
            "delta service definition",
        ]

    def test_context_does_not_cover_returns_none(self):
        text = "Answer: Context does not cover the requested information."
        assert extract_choice(text, self._generic_choices()) is None

    def test_context_does_not_specify_returns_none(self):
        text = "Answer: Context does not specify the exact value."
        assert extract_choice(text, self._generic_choices()) is None

    def test_none_of_options_returns_none(self):
        text = "None of the options can be confirmed based on the given chunks."
        assert extract_choice(text, self._generic_choices()) is None

    def test_cannot_determine_returns_none(self):
        text = "Cannot determine the answer from the provided context."
        assert extract_choice(text, self._generic_choices()) is None

    def test_insufficient_context_returns_none(self):
        text = "Insufficient context to provide a definitive answer."
        assert extract_choice(text, self._generic_choices()) is None

    def test_explicit_letter_beats_refusal(self):
        # If model gives a letter THEN says context-poor, the letter still wins.
        text = "Answer: B\n\nNote: context does not cover the full picture but B is closest."
        assert extract_choice(text, self._generic_choices()) == 1

    def test_refusal_in_middle_does_not_block_letter(self):
        # Refusal phrase later in the text shouldn't block an early letter answer.
        text = (
            "Answer: C\n\nJustification: While context does not specify every "
            "detail, option C aligns best with the available evidence."
        )
        assert extract_choice(text, self._generic_choices()) == 2


# ---------- _is_refusal helper ---------------------------------------------


class TestIsRefusal:
    def test_detects_first_line_refusal(self):
        assert _is_refusal("Context does not cover this question")

    def test_does_not_flag_mid_text_refusal(self):
        text = "Answer: B\n\nContext does not cover X but B is correct."
        # The headline is "Answer: B", so even though refusal appears later
        # in line 1 (joined by spaces), it should detect → but extract_choice
        # already returned at pattern 2. Helper itself only inspects line 1.
        # Test: line 1 is just "Answer: B" — should NOT be flagged.
        assert not _is_refusal(text)

    def test_empty_text(self):
        assert not _is_refusal("")
        assert not _is_refusal(None)  # type: ignore[arg-type]


# ---------- Regression cases from the actual benchmark run -----------------


class TestRealRegressions:
    """The 7 real refusals observed in
    `origin_rerank_100_question_deepseek_fix_cypher_gen_think`. All MUST
    return None (extraction failure) — never a hallucinated pick."""

    def test_teleaction_refusal(self):
        text = (
            'The context provided does not include any specific information or '
            'references related to "teleaction service." None of the options can '
            'be confirmed based on the given chunks, as they do not address '
            'teleaction services.\n\n'
            'Answer: Context does not cover the definition of a teleaction service.'
        )
        choices = [
            "A telecommunication service that uses short messages requiring a low transmission rate",
            "A service offered by a PLMN operator",
            "A type of telecommunication service that provides complete capability",
            "A type of telecommunication service that uses text conversation",
        ]
        assert extract_choice(text, choices) is None

    def test_mac_offline_refusal(self):
        text = "Context does not cover Which of the following is a limitation of applying offline solutions in MAC (Medium Access Control) protocols?"
        choices = ["Increased packet loss", "Increased computational complexity",
                   "Increased energy consumption", "Increased transmission overhead"]
        assert extract_choice(text, choices) is None

    def test_cell_id_refusal(self):
        text = "The context does not specify the capacity of Cell-ID in an eNB.\n\nAnswer: Context does not cover the capacity of Cell-ID in an eNB for 3GPP Release 14."
        choices = ["1.04 million", "2.04 million", "1024", "256"]
        assert extract_choice(text, choices) is None

    def test_pap_refusal_with_markdown(self):
        text = (
            "Answer: Context does not cover what the Policy Administration Point (PAP) does.\n\n"
            "**Justification:** The provided context does not include any information about the PAP."
        )
        choices = ["Power distribution in data centers", "Management of network policies",
                   "Physical infrastructure management", "Evaluation of server efficiency rating",
                   "Standard performance evaluation"]
        assert extract_choice(text, choices) is None
