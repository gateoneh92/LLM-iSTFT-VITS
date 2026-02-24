"""
IPA (International Phonetic Alphabet) Tokenizer
모든 언어를 발음 기호로 변환하여 학습
"""

# IPA 심볼 정의 (주요 음소들)
IPA_SYMBOLS = [
    # Padding and special tokens
    '<pad>', '<sos>', '<eos>',

    # Vowels (모음)
    'i', 'y', 'ɨ', 'ʉ', 'ɯ', 'u',
    'ɪ', 'ʏ', 'ʊ',
    'e', 'ø', 'ɘ', 'ɵ', 'ɤ', 'o',
    'ə', 'ɚ', 'ᵻ',  # r-colored and reduced vowels
    'ɛ', 'œ', 'ɜ', 'ɞ', 'ʌ', 'ɔ',
    'æ', 'ɐ',
    'a', 'ɶ', 'ɑ', 'ɒ',

    # Consonants (자음)
    # Plosives
    'p', 'b', 't', 'd', 'ʈ', 'ɖ', 'c', 'ɟ', 'k', 'g', 'ɡ', 'q', 'ɢ', 'ʔ',

    # Nasals
    'm', 'ɱ', 'n', 'ɳ', 'ɲ', 'ŋ', 'ɴ',

    # Trills
    'ʙ', 'r', 'ʀ',

    # Taps/Flaps
    'ⱱ', 'ɾ', 'ɽ',

    # Fricatives
    'ɸ', 'β', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'ʂ', 'ʐ', 'ç', 'ʝ', 'x', 'ɣ', 'χ', 'ʁ', 'ħ', 'ʕ', 'h', 'ɦ',

    # Lateral fricatives
    'ɬ', 'ɮ',

    # Approximants
    'ʋ', 'ɹ', 'ɻ', 'j', 'ɰ', 'w',

    # Lateral approximants
    'l', 'ɭ', 'ʎ', 'ʟ',

    # Affricates (일반적인 것들)
    'ʦ', 'ʣ', 'ʧ', 'ʤ',

    # Suprasegmentals (운율)
    'ˈ',  # Primary stress
    'ˌ',  # Secondary stress
    'ː',  # Long
    'ˑ',  # Half-long
    '.',  # Syllable break
    '|',  # Minor group
    '‖',  # Major group

    # Tones (성조)
    '˥', '˦', '˧', '˨', '˩',  # Level tones
    '↗', '↘',  # Rising, falling

    # Diacritics (주요)
    'ʰ', 'ʷ', 'ʲ', 'ˠ', 'ˤ',  # Secondary articulation
    '̃', '̩', '̯',  # Nasalized, syllabic, non-syllabic

    # Space and punctuation
    ' ', ',', '.', '!', '?', ';', '"', '"', ':', '(', ')',
]

# Vocab to index mapping
IPA_VOCAB = {symbol: idx for idx, symbol in enumerate(IPA_SYMBOLS)}
IPA_VOCAB_SIZE = len(IPA_SYMBOLS)

print(f"IPA Vocabulary size: {IPA_VOCAB_SIZE}")


class IPATokenizer:
    """Simple character-level IPA tokenizer"""

    def __init__(self):
        self.vocab = IPA_VOCAB
        self.idx_to_symbol = {idx: symbol for symbol, idx in IPA_VOCAB.items()}
        self.pad_token_id = self.vocab['<pad>']
        self.sos_token_id = self.vocab['<sos>']
        self.eos_token_id = self.vocab['<eos>']

    def encode(self, text, add_special_tokens=True):
        """
        IPA 텍스트를 인덱스로 변환

        Args:
            text: IPA string (e.g., "həˈloʊ")
            add_special_tokens: Add <sos> and <eos>

        Returns:
            List of token indices
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.sos_token_id)

        i = 0
        while i < len(text):
            # Try to match longest symbol first (for diacritics)
            matched = False
            for length in [3, 2, 1]:  # Try 3-char, 2-char, 1-char
                if i + length <= len(text):
                    symbol = text[i:i+length]
                    if symbol in self.vocab:
                        tokens.append(self.vocab[symbol])
                        i += length
                        matched = True
                        break

            if not matched:
                # Unknown symbol, use space or skip
                print(f"Warning: Unknown IPA symbol '{text[i]}', skipping")
                i += 1

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, token_ids):
        """
        인덱스를 IPA 텍스트로 변환

        Args:
            token_ids: List of token indices

        Returns:
            IPA string
        """
        symbols = []
        for idx in token_ids:
            if idx in self.idx_to_symbol:
                symbol = self.idx_to_symbol[idx]
                if symbol not in ['<pad>', '<sos>', '<eos>']:
                    symbols.append(symbol)

        return ''.join(symbols)

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        """Huggingface-style interface"""
        import torch

        tokens = self.encode(text, add_special_tokens=True)

        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        if return_tensors == 'pt':
            input_ids = torch.tensor([tokens])
            return {'input_ids': input_ids}

        return {'input_ids': tokens}


# 영어 → IPA 변환 예제 (실제로는 phonemizer 라이브러리 사용 권장)
ENGLISH_TO_IPA = {
    'hello': 'həˈloʊ',
    'world': 'wɜːld',
    'test': 'tɛst',
    'number': 'ˈnʌmbər',
    'one': 'wʌn',
    'two': 'tuː',
    'three': 'θriː',
}


def text_to_ipa(text, language='en'):
    """
    텍스트를 IPA로 변환

    실제 사용 시에는 phonemizer 라이브러리 사용 권장:
    from phonemizer import phonemize
    ipa = phonemize(text, language='en-us', backend='espeak')
    """
    words = text.lower().split()
    ipa_words = []

    for word in words:
        if word in ENGLISH_TO_IPA:
            ipa_words.append(ENGLISH_TO_IPA[word])
        else:
            # Fallback: use original text
            ipa_words.append(word)

    return ' '.join(ipa_words)


if __name__ == "__main__":
    # Test
    tokenizer = IPATokenizer()

    # Test encoding
    ipa_text = "həˈloʊ wɜːld"
    tokens = tokenizer.encode(ipa_text)
    print(f"IPA text: {ipa_text}")
    print(f"Tokens: {tokens}")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    # Test English to IPA
    english_text = "hello world"
    ipa = text_to_ipa(english_text)
    print(f"\nEnglish: {english_text}")
    print(f"IPA: {ipa}")
