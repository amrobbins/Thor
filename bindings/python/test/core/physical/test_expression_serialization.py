import json

import thor
from thor.physical import DynamicExpression
from thor.physical import Expression as ex
from thor.physical import ExpressionDefinition
from thor.physical import Outputs


def test_outputs_round_trip_through_expression_definition_json():
    x = ex.input("x", compute_dtype=thor.DataType.fp32)
    y = ex.input("y", output_dtype=thor.DataType.fp16)
    scale = ex.input("scale")

    outputs = ex.outputs({
        "sum": x + y,
        "scaled": ex.reduce_sum(x * scale, axis=1, squeeze=True),
    })

    payload = outputs.to_json()
    payload_json = json.loads(payload)

    assert payload_json["type"] == "thor.expression"
    assert payload_json["schema_version"] == 1
    assert payload_json["expected_output_names"] == ["sum", "scaled"]
    assert payload_json["canonical_hash"].startswith("fnv1a64:")

    definition = ExpressionDefinition.from_json(payload)
    assert definition.expected_input_names == ["x", "y", "scale"]
    assert definition.expected_output_names == ["sum", "scaled"]
    assert definition.canonical_hash.startswith("fnv1a64:")

    restored_outputs = Outputs.from_json(payload)
    assert restored_outputs.output_names() == ["sum", "scaled"]

    dynamic_expression = DynamicExpression.from_expression_definition(definition)
    assert dynamic_expression.serialized_definition is not None
    assert json.loads(dynamic_expression.serialized_definition.to_json()) == payload_json
