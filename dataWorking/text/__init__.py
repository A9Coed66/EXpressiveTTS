# """ from https://github.com/keithito/tacotron """

# import re
# from text import cleaners
# from text.symbols import symbols


# _symbol_to_id = {s: i for i, s in enumerate(symbols)}
# _id_to_symbol = {i: s for i, s in enumerate(symbols)}

# #_symbol_to_id: {'_': 0, '-': 1, '!': 2, "'": 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, ';': 9, '?': 10, ' ': 11, 
# # 'A': 12, 'B': 13, 'C': 14, 'D': 15, 'E': 16, 'F': 17, 'G': 18, 'H': 19, 'I': 20, 'J': 21, 'K': 22, 'L': 23, 'M': 24, 
# # 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 33, 'W': 34, 'X': 35, 'Y': 36, 'Z': 37, 
# # 'a': 38, 'b': 39, 'c': 40, 'd': 41, 'e': 42, 'f': 43, 'g': 44, 'h': 45, 'i': 46, 'j': 47, 'k': 48, 'l': 49, 'm': 50, 
# # 'n': 51, 'o': 52, 'p': 53, 'q': 54, 'r': 55, 's': 56, 't': 57, 'u': 58, 'v': 59, 'w': 60, 'x': 61, 'y': 62, 'z': 63, 
# # '@AA': 64, '@AA0': 65, '@AA1': 66, '@AA2': 67, '@AE': 68, '@AE0': 69, '@AE1': 70, '@AE2': 71, '@AH': 72, '@AH0': 73, 
# # '@AH1': 74, '@AH2': 75, '@AO': 76, '@AO0': 77, '@AO1': 78, '@AO2': 79, '@AW': 80, '@AW0': 81, '@AW1': 82, '@AW2': 83, 
# # '@AY': 84, '@AY0': 85, '@AY1': 86, '@AY2': 87, '@B': 88, '@CH': 89, '@D': 90, '@DH': 91, '@EH': 92, '@EH0': 93, 
# # '@EH1': 94, '@EH2': 95, '@ER': 96, '@ER0': 97, '@ER1': 98, '@ER2': 99, '@EY': 100, '@EY0': 101, '@EY1': 102, 
# # '@EY2': 103, '@F': 104, '@G': 105, '@HH': 106, '@IH': 107, '@IH0': 108, '@IH1': 109, '@IH2': 110, '@IY': 111, 
# # '@IY0': 112, '@IY1': 113, '@IY2': 114, '@JH': 115, '@K': 116, '@L': 117, '@M': 118, '@N': 119, '@NG': 120, '@OW': 121, 
# # '@OW0': 122, '@OW1': 123, '@OW2': 124, '@OY': 125, '@OY0': 126, '@OY1': 127, '@OY2': 128, '@P': 129, '@R': 130, '@S': 131, 
# # '@SH': 132, '@T': 133, '@TH': 134, '@UH': 135, '@UH0': 136, '@UH1': 137, '@UH2': 138, '@UW': 139, '@UW0': 140, '@UW1': 141, 
# # '@UW2': 142, '@V': 143, '@W': 144, '@Y': 145, '@Z': 146, '@ZH': 147}

# _curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


# def get_arpabet(word, dictionary):
#     word_arpabet = dictionary.lookup(word)
#     if word_arpabet is not None:
#         return "{" + word_arpabet[0] + "}"
#     else:
#         return word


# def text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=None):
#     '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

#     The text can optionally have ARPAbet sequences enclosed in curly braces embedded
#     in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

#     Args:
#       text: string to convert to a sequence
#       cleaner_names: names of the cleaner functions to run the text through
#       dictionary: arpabet class with arpabet dictionary

#     Returns:
#       List of integers corresponding to the symbols in the text
#     '''
#     sequence = []
#     space = _symbols_to_sequence(' ')
#     # print(f'space: {space}')
#     # Check for curly braces and treat their contents as ARPAbet:
#     while len(text):
#         m = _curly_re.match(text)
#         # print(f'm: {m}')
#         if not m:
#             clean_text = _clean_text(text, cleaner_names)
#             # print(f'clean_text: {clean_text}')
#             if dictionary is not None:
#                 clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
#                 # print(f'clean_text after dictionary: {clean_text}')
#                 for i in range(len(clean_text)):
#                     t = clean_text[i]
#                     # print(f'clean_text[i]: {t}')
#                     if t.startswith("{"):
#                         sequence += _arpabet_to_sequence(t[1:-1])
#                     else:
#                         sequence += _symbols_to_sequence(t)
#                     sequence += space
#             else:
#                 sequence += _symbols_to_sequence(clean_text)
#             break
#         sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
#         sequence += _arpabet_to_sequence(m.group(2))
#         text = m.group(3)
  
#     # remove trailing space
#     if dictionary is not None:
#         sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
#     return sequence


# def sequence_to_text(sequence):
#     '''Converts a sequence of IDs back to a string'''
#     result = ''
#     for symbol_id in sequence:
#         if symbol_id in _id_to_symbol:
#             s = _id_to_symbol[symbol_id]
#             # Enclose ARPAbet back in curly braces:
#             if len(s) > 1 and s[0] == '@':
#                 s = '{%s}' % s[1:]
#             result += s
#     return result.replace('}{', ' ')


# def _clean_text(text, cleaner_names):
#     # from text to phoneme if possible
#     for name in cleaner_names:
#         cleaner = getattr(cleaners, name)
#         if not cleaner:
#             raise Exception('Unknown cleaner: %s' % name)
#         text = cleaner(text)
#     return text


# def _symbols_to_sequence(symbols):
#     return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


# def _arpabet_to_sequence(text):
#     return _symbols_to_sequence(['@' + s for s in text.split()])


# def _should_keep_symbol(s):
#     return s in _symbol_to_id and s != '_' and s != '~'
""" from https://github.com/keithito/tacotron """

import re
from text import cleaners
from text.symbols import symbols
import sys

print(f'symbols: {symbols}')
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

#_symbol_to_id: {'_': 0, '-': 1, '!': 2, "'": 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, ';': 9, '?': 10, ' ': 11, 
# 'A': 12, 'B': 13, 'C': 14, 'D': 15, 'E': 16, 'F': 17, 'G': 18, 'H': 19, 'I': 20, 'J': 21, 'K': 22, 'L': 23, 'M': 24, 
# 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32, 'V': 33, 'W': 34, 'X': 35, 'Y': 36, 'Z': 37, 
# 'a': 38, 'b': 39, 'c': 40, 'd': 41, 'e': 42, 'f': 43, 'g': 44, 'h': 45, 'i': 46, 'j': 47, 'k': 48, 'l': 49, 'm': 50, 
# 'n': 51, 'o': 52, 'p': 53, 'q': 54, 'r': 55, 's': 56, 't': 57, 'u': 58, 'v': 59, 'w': 60, 'x': 61, 'y': 62, 'z': 63, 
# '@AA': 64, '@AA0': 65, '@AA1': 66, '@AA2': 67, '@AE': 68, '@AE0': 69, '@AE1': 70, '@AE2': 71, '@AH': 72, '@AH0': 73, 
# '@AH1': 74, '@AH2': 75, '@AO': 76, '@AO0': 77, '@AO1': 78, '@AO2': 79, '@AW': 80, '@AW0': 81, '@AW1': 82, '@AW2': 83, 
# '@AY': 84, '@AY0': 85, '@AY1': 86, '@AY2': 87, '@B': 88, '@CH': 89, '@D': 90, '@DH': 91, '@EH': 92, '@EH0': 93, 
# '@EH1': 94, '@EH2': 95, '@ER': 96, '@ER0': 97, '@ER1': 98, '@ER2': 99, '@EY': 100, '@EY0': 101, '@EY1': 102, 
# '@EY2': 103, '@F': 104, '@G': 105, '@HH': 106, '@IH': 107, '@IH0': 108, '@IH1': 109, '@IH2': 110, '@IY': 111, 
# '@IY0': 112, '@IY1': 113, '@IY2': 114, '@JH': 115, '@K': 116, '@L': 117, '@M': 118, '@N': 119, '@NG': 120, '@OW': 121, 
# '@OW0': 122, '@OW1': 123, '@OW2': 124, '@OY': 125, '@OY0': 126, '@OY1': 127, '@OY2': 128, '@P': 129, '@R': 130, '@S': 131, 
# '@SH': 132, '@T': 133, '@TH': 134, '@UH': 135, '@UH0': 136, '@UH1': 137, '@UH2': 138, '@UW': 139, '@UW0': 140, '@UW1': 141, 
# '@UW2': 142, '@V': 143, '@W': 144, '@Y': 145, '@Z': 146, '@ZH': 147}

_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')
print(f'_curly_re: {_curly_re}')


def get_arpabet(word, dictionary):
    word_arpabet = dictionary.lookup(word)
    if word_arpabet is not None:
        return "{" + word_arpabet[0] + "}"
    else:
        return word


def text_to_sequence(text, cleaner_names=["vietnamese_cleaners"], dictionary=None):
    #TODO: 1. text to IPA, 2. IPA to x-sampa, 3. x-sampa to sequence
    '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      dictionary: arpabet class with arpabet dictionary

    Returns:
      List of integers corresponding to the symbols in the text
    '''
    sequence = []
    space = _symbols_to_sequence(' ')
    # print(f'space: {space}')
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        # print(f'm: {m}')
        if not m:
            clean_text = _clean_text(text, cleaner_names)
            if dictionary is not None:
                clean_text = [get_arpabet(w, dictionary) for w in clean_text.split(" ")]
                for i in range(len(clean_text)):
                    t = clean_text[i]
                    # print(f'clean_text[i]: {t}')
                    if t.startswith("{"):
                        sequence += _arpabet_to_sequence(t[1:-1])
                    else:
                        sequence += _symbols_to_sequence(t)
                    sequence += space
            else:
                sequence += _symbols_to_sequence(clean_text)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)
  
    # remove trailing space
    if dictionary is not None:
        sequence = sequence[:-1] if sequence[-1] == space[0] else sequence
    return sequence


def sequence_to_text(sequence):
    '''Converts a sequence of IDs back to a string'''
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
    # from text to phoneme if possible
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id
