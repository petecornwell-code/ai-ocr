from app.service.crew_service import tally_votes


class TestTallyVotesUnanimous:
    def test_all_three_agree(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": "John", "amount": "$100"},
                "agent_b": {"name": "John", "amount": "$100"},
                "agent_c": {"name": "John", "amount": "$100"},
            },
            field_names=["name", "amount"],
        )
        assert result["summary"]["consensus_count"] == 2
        assert result["summary"]["intervention_count"] == 0
        assert result["fields"]["name"]["status"] == "consensus"
        assert result["fields"]["name"]["value"] == "John"
        assert result["fields"]["amount"]["value"] == "$100"


class TestTallyVotesMajority:
    def test_two_of_three_agree(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": "John", "amount": "$100"},
                "agent_b": {"name": "John", "amount": "$200"},
                "agent_c": {"name": "Jon", "amount": "$100"},
            },
            field_names=["name", "amount"],
        )
        assert result["fields"]["name"]["status"] == "consensus"
        assert result["fields"]["name"]["value"] == "John"
        assert result["fields"]["amount"]["status"] == "consensus"
        assert result["fields"]["amount"]["value"] == "$100"
        assert result["summary"]["consensus_count"] == 2
        assert result["summary"]["intervention_count"] == 0

    def test_different_fields_have_different_outcomes(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": "John", "date": "2024-01-01"},
                "agent_b": {"name": "John", "date": "2024-01-02"},
                "agent_c": {"name": "John", "date": "2024-01-03"},
            },
            field_names=["name", "date"],
        )
        assert result["fields"]["name"]["status"] == "consensus"
        assert result["fields"]["date"]["status"] == "intervention_required"
        assert result["summary"]["consensus_count"] == 1
        assert result["summary"]["intervention_count"] == 1


class TestTallyVotesIntervention:
    def test_all_three_disagree(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": "John"},
                "agent_b": {"name": "Jon"},
                "agent_c": {"name": "Joan"},
            },
            field_names=["name"],
        )
        assert result["fields"]["name"]["status"] == "intervention_required"
        assert result["fields"]["name"]["value"] is None
        assert result["summary"]["intervention_count"] == 1
        assert result["summary"]["consensus_count"] == 0

    def test_all_fields_need_intervention(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"f1": "a", "f2": "x"},
                "agent_b": {"f1": "b", "f2": "y"},
                "agent_c": {"f1": "c", "f2": "z"},
            },
            field_names=["f1", "f2"],
        )
        assert result["summary"]["consensus_count"] == 0
        assert result["summary"]["intervention_count"] == 2


class TestTallyVotesEdgeCases:
    def test_null_values_count_as_agreement(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": None},
                "agent_b": {"name": None},
                "agent_c": {"name": "John"},
            },
            field_names=["name"],
        )
        assert result["fields"]["name"]["status"] == "consensus"
        assert result["fields"]["name"]["value"] is None

    def test_missing_field_treated_as_none(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": "John"},
                "agent_b": {},
                "agent_c": {},
            },
            field_names=["name"],
        )
        assert result["fields"]["name"]["status"] == "consensus"
        assert result["fields"]["name"]["value"] is None

    def test_votes_preserved_in_output(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"name": "John"},
                "agent_b": {"name": "John"},
                "agent_c": {"name": "Jon"},
            },
            field_names=["name"],
        )
        votes = result["fields"]["name"]["votes"]
        assert votes["agent_a"] == "John"
        assert votes["agent_b"] == "John"
        assert votes["agent_c"] == "Jon"

    def test_single_field(self):
        result = tally_votes(
            agent_results={
                "agent_a": {"x": "1"},
                "agent_b": {"x": "1"},
                "agent_c": {"x": "2"},
            },
            field_names=["x"],
        )
        assert result["summary"]["total_fields"] == 1
        assert result["fields"]["x"]["value"] == "1"

    def test_many_fields(self):
        fields = {f"f{i}": str(i) for i in range(10)}
        result = tally_votes(
            agent_results={
                "agent_a": fields,
                "agent_b": fields,
                "agent_c": fields,
            },
            field_names=list(fields.keys()),
        )
        assert result["summary"]["total_fields"] == 10
        assert result["summary"]["consensus_count"] == 10
        assert result["summary"]["intervention_count"] == 0
