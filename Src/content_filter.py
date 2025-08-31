from utils import simple_clean
import re

def age_appropriate(content: str, user_age: int):
    """
    Simple rules:
     - If user_age < 13: block explicit mature keywords
     - If 13 <= user_age < 16: apply softer threshold
     - Over 16: allow
    Returns: dict { 'action': 'allow'|'block'|'flag', 'reason': str }
    """
    text = simple_clean(content)
    explicit_keywords = ['kill', 'suicide', 'porn', 'sex', 'rape']
    sexual_words = ['porn', 'sex', 'rape']
    violence_words = ['kill', 'murder', 'stab', 'shoot']

    found_explicit = [w for w in explicit_keywords if w in text]
    if user_age < 13:
        if found_explicit:
            return {'action':'block', 'reason': f'explicit content: {found_explicit}'}
        return {'action':'allow', 'reason':'age ok'}
    elif user_age < 16:
        if any(w in text for w in sexual_words):
            return {'action':'block', 'reason':'sexual content not appropriate'}
        if any(w in text for w in violence_words):
            return {'action':'flag', 'reason':'violent content; requires review'}
        return {'action':'allow', 'reason':'age ok'}
    else:
        # for older users, only flag severe self-harm
        if 'suicide' in text or 'kill myself' in text:
            return {'action':'flag', 'reason':'self-harm content'}
        return {'action':'allow', 'reason':'age ok'}
