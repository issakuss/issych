def alphabet_to_num(alphabet: str):
    """
    >> alphabet_to_num('A')
    1
    >> alphabet_to_num('a')
    1
    >> alphabet_to_num('Z')
    26
    >> alphabet_to_num('AA')
    27
    """
    N_ALPHABET = 26
    num = 0
    alphabet = alphabet.upper()
    for i, item in enumerate(alphabet):
        order = (ord(item) - ord('A') + 1)
        num += order * pow(N_ALPHABET, len(alphabet) - i - 1)
    return num
