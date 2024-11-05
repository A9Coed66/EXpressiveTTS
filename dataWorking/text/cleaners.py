""" from https://github.com/keithito/tacotron """

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from phonemizer.backend import EspeakBackend 
from phonecodes import phonecodes



_whitespace_re = re.compile(r'\s+')

_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text
#---------------------------------------------------------
# Add Vietnamese cleaner

language = "vie"

if language == "en":
    g2p_lang = "en-us"  # English as spoken in USA
    expand_abbreviations = lambda x: x
    phonemizer = "espeak"
elif language == "vie":
    g2p_lang = "vi"  # Northern Vietnamese
    expand_abbreviations = lambda x: x
    phonemizer = "espeak"

elif language == "vi-ctr":
    g2p_lang = "vi-vn-x-central"  # Central Vietnamese
    expand_abbreviations = lambda x: x
    phonemizer = "espeak"

elif language == "vi-so":
    g2p_lang = "vi-vn-x-south"  # Southern Vietnamese
    expand_abbreviations = lambda x: x
    phonemizer = "espeak"

phonemizer_backend = EspeakBackend(language=g2p_lang,
                                    punctuation_marks='*;:,.!?¡¿—…()"«»“”~/。【】、‥،؟“”؛',
                                    preserve_punctuation=True,
                                    language_switch='remove-flags',
                                    with_stress=False)

def vietnamese_cleaners(text):
    text = lowercase(text)
    phones = phonemizer_backend.phonemize([text], strip=True)[0]
    text = phonecodes.convert(phones, "ipa", "xsampa")
    text = collapse_whitespace(text)
    return text