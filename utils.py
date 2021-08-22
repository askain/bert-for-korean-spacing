from typing import List

def load_slot_labels() -> List[str]:
    """
    tag label 종류 리턴

    - UNK: Unknown
    - PAD: ...
    - O: ????
    - B: Begin
    - I: Inside
    - E: End
    - S: Single
    """
    return ["UNK", "PAD", "O", "B", "I", "E", "S"]