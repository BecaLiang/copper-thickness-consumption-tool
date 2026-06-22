def deep_merge_two_dicts(left_dict: dict, right_dict: dict) -> dict:
    assert isinstance(left_dict, dict) and isinstance(right_dict, dict)
    output_dict = dict(left_dict) # shallow copy of left_dict
    for key, value in right_dict.items():
        if (
            key in output_dict
            and isinstance(output_dict[key], dict)
            and isinstance(value, dict)
        ):
            output_dict[key] = deep_merge_two_dicts(output_dict[key], value)
        else:
            output_dict[key] = value
    return output_dict


def round_to_point_five(X: float) -> float:
    return round(X * 2) / 2