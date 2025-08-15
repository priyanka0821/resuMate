import spacy
import re
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    tokens = re.findall(r'\b[a-zA-Z0-9\-\+\.#]+\b', text)
    return list(set(tokens))



def extract_phrases(text):
    doc = nlp(text.lower())
    phrases = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1]
    return phrases
