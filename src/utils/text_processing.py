import re

clinical = ["General-Clinical", "Clinical-Surgery", "Talk-Very-Fast-Clinical", 
            "Pre-Clinical", "Clinical-Medicine", "Pre-Clinical-INT"]
general = ["40yrs-old-and-above", "Talk-Very-Fast-Anyone", 'Transcribe-Inference',
           "Naija-News-Non-Clinical", "News-Anyone-INT", 'Transcribe-Conversation']
legal = ['Legally-Speaking', 'Transcribe-NASS', 'Transcribe-Kenya', 
         'Transcribe-South-Africa', 'Transcribe-Ghana']

inaudible_tags = ['[music] [inaudible]', '(inaudible) ', '[inaudible)', '(inaudible]',
                  '[Inaudible].', '[music]','[INAUDIBLE]',' [Inaudible]', '(Inaudible).',
                  '[Inaudible] ', '[silence]','[Silence]', '[inaudible] ', 'in aduible',
                  '(inaudible)','(Inaudible)','[Inaudible]', 'Inaudible','[inaudible]',
                  '[inaudable]','[Inaudible]','Inaudable ','Blank ', 'inaudible', 'Inaudible ', 
                  '(audio is empty)', 'noise', '(noise)', '[noise]', 'Blank'
                 ]
inaudible_tags_regex = [x.replace('[', '\[').replace(']', '\]').replace('(', '\(').replace(')', '\)') for x in inaudible_tags]
inaudible_tags_joined = "|".join(inaudible_tags_regex)
rx = re.compile(inaudible_tags_joined, re.I)


def clean_text(text):
    """
    post processing to normalized reference and predicted transcripts
    :param text: str
    :return: str
    """
    # remove multiple spaces
    if type(text) != str:
        print(text)
        return " "
    text = re.sub(r"\s\s+", " ", text)
    # strip trailing spaces
    text = text.strip()
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = text.lower()
    text = text.replace(" comma,", ",") \
        .replace(" koma,", " ") \
        .replace(" coma,", " ") \
        .replace(" full stop.", ".") \
        .replace(" full stop", ".") \
        .replace(",.", ".") \
        .replace(",,", ",") \
        .strip()
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    return text


def detect_inaudible(text):
    if (text in inaudible_tags) or (text.strip().lower() in ['inaudible', '[inaudible]', '(inaudible)']):
        return 1
    elif rx.search(text):
        return 2
    return 0


def replace_inaudible(text, pad_token=''):
    if (text in inaudible_tags) or (text.strip().lower() in ['inaudible', '[inaudible]', '(inaudible)']):
        text = pad_token
    else:
        text = re.sub(inaudible_tags_joined, pad_token, text)

    text = text.replace('[', '').replace(']', '')
    return text


text_to_digit = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": '00',
    "thousand": '000',
    "million": '000000',
    "billion": '000000000',
    "first": '1st',
    "second": '2nd',
    "third": '3rd',
    "fourth": '4th',
    "fifth": '5th',
    "sixth": '6th',
    "seventh": '7th',
    "eighth": '8th',
    "nineth": '9th',
    "tenth": '10th',
    "eleventh": '11th',
    "twelveth": '12th',
    "thirteenth": '13th',
    "fourteenth": '14th',
    "fifteenth": '15th',
    "sixteenth": '16th',
    "seventeenth": '17th',
    "eighteenth": '18th',
    "nineteenth": '19th',
    "twentieth": '20th',
    "thirtieth": '30th',
    "fortieth": '40th',
    "fiftieth": '50th',
    "sixtieth": '60th',
    "seventieth": '70th',
    "eightieth": '80th',
    "ninetieth": '90th',
    "hundredth": '00th',
    "thousandth": '000th',
    "millionth": '000000th',
    "billionth": '000000000th',
}


def text_to_numbers(text):
    text = text.split()
    return " ".join([str(text_to_digit[digit.lower()]) if digit.lower() in text_to_digit else digit for digit in text])


def strip_task_tags(s):
    if s.endswith('>'):
        return s[:s.find('<')]
    elif s.startswith('<'):
        return s[s.rfind('>')+1:]
    return s


def get_task_tags(s):
    if s.endswith('>'):
        return s[s.find('<'):]
    elif s.startswith('<'):
        return s[:s.rfind('>')+1]
    return s


def assign_domain(project_name):
    if project_name in clinical:
        return "clinical"
    elif project_name in general:
        return "general"
    elif project_name in legal:
        return "legal"
    else:
        return "general"
    
def is_accent_multiple(s):
    #print(f"accent:--{s}--")
    if len(s.split('_')) > 2:
        return 1
    elif 'pair_' in s:
        return 1
    else:
        return 0
    

def get_minority_accents(data, majority_count=5000):
    accent_counts = data.accent.value_counts().to_dict()
    print(accent_counts)
    return [accent for accent, count in accent_counts.items() if count < majority_count]
    
    