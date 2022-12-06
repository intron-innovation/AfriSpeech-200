import re

remove_chars = '[\ṅ\_\\t\@\~\;\“\"\‘\”\#\&\*\=\^\{\}\|\é\\x13\…\ü\ß\ë\!\ó\ç\√\ï\–\—\u202f\u3000\u202f\u2009'
remove_chars += '\xa0\x01\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x17$¡¢¤¥¦§¨©ª«¬¯±²³´µ'
remove_chars += 'º¼½¾ÉÖ×àçéíö÷üıŁłŒœŠšŸŽžƒˆˇ˘˙˚˛˜˝–—‘’‚“”„†‡•…‰‹›⁄€−îíéèçßÞÕÔÓÒÐ»°£¡ÅÚ†'
remove_chars += '\`\’\á\ć\ñ\α\ū\ú\ã\ě\ä\à\ā\ʻ\ř\í\è\å\â\ạ\ø\¡\ô\ō\ő\ō\ö\ö\x02\x03¡¥¨©®°´µÐÒÓÔÕ×ÞßıŒ˚˜–‚”†™ﬁﬂ]'


def clean_text(text):
    text = re.sub(remove_chars, '', text)
    text = text.replace('>', '')
    text = text.replace('\t', ' ')
    text = text.replace('\n', '')
    text = " ".join(text.split())
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\-\?\:\'\/\(\)\[\]\+\%]", '', text)
    text = text.lower()
    return text
