import numpy as np

from parallax3d.segmentation import _Candidate, select_layers


def test_selection_keeps_clean_objects_and_gives_sky_independent_motion() -> None:
    height, width = 100, 160
    sky = np.zeros((height, width), dtype=bool)
    sky[:30] = True
    person = np.zeros((height, width), dtype=bool)
    person[30:92, 55:95] = True
    person_duplicate = person.copy()
    person_duplicate[31:91, 56:94] = True

    layers = select_layers(
        [
            _Candidate("sky", 0.97, sky, "semantic"),
            _Candidate("person", 0.96, person, "sam2"),
            _Candidate("person", 0.82, person_duplicate, "sam2"),
        ],
        maximum_layers=6,
    )

    assert len(layers) == 2
    assert layers[0].label == "sky"
    assert layers[0].motion == "sky"
    assert layers[0].depth < layers[1].depth
    assert layers[1].erase_from_background
    assert layers[1].mask.dtype == np.uint8


def test_selection_fills_remaining_slots_without_comparing_numpy_masks() -> None:
    height, width = 100, 160
    candidates = []
    for index in range(4):
        mask = np.zeros((height, width), dtype=bool)
        left = 8 + index * 35
        mask[40:90, left : left + 20] = True
        candidates.append(_Candidate("object", 0.9, mask, "sam2"))

    layers = select_layers(candidates, maximum_layers=4)

    assert len(layers) == 4
