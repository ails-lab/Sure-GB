import spacy
from spacy.matcher import Matcher
import numpy as np

class English:
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md") 
        self.nlp.add_pipe('coreferee')
            
class French:
    
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_md") 
        self.nlp.add_pipe('coreferee')
        
class Greek: 
    
    def __init__(self):
        self.nlp = spacy.load("el_core_news_md") 
    
    
def find_gender(nlp, text, extracted_occups, check_coreference = True):
    
    doc = nlp(text)
    
    for v in extracted_occups:
        occ = v["title"]
        
        resp = []
        for [_, ref_start, ref_end] in link_occupation_mentions_with_lemmatization(nlp, text, occ):
            
            gender_pronoun, prob_pronoun = find_gender_using_pronoun(doc, [ref_start, ref_end])
            
            gender_coref, prob_coref = "NC", 0.5
            if check_coreference:
                gender_coref, prob_coref = find_gender_using_coreference(doc, [ref_start, ref_end], explain=False)
            
            gender_word, prob_word = find_gender_using_word(doc, [ref_start, ref_end])
            
            label = None
            t = None
            if gender_pronoun != "NC":
                label = gender_pronoun
                t = "pronoun"
            elif gender_coref != "NC":
                label = gender_coref
                t = "coref"
            elif gender_word != "NC":
                label = gender_word
                t = "word"
            else:
                label = "NC"
            
            if len(doc) == ref_end:
                end_ind = doc[ref_start].idx + len(str(doc[ref_start]))
            else:
                end_ind = doc[ref_end].idx
                
            resp.append({"tokens": [ref_start, ref_end], 
                         "indexes": [doc[ref_start].idx, end_ind],
                        "gender": label,
                        "method": t}) 
    return resp

    
def link_occupation_mentions_with_lemmatization(nlp, text, occupation):        
    doc = nlp(text.lower())
    matcher = Matcher(nlp.vocab)
    
    # Setup the patterns using lemmatization
    occupation = occupation.lower()
    occupation_doc = nlp(occupation)  # Process the occupation term
    # Create a pattern based on the lemmas of the occupation phrase
    pattern = [{"LEMMA": token.lemma_} for token in occupation_doc]
    matcher.add("OCCUPATION", [pattern])
    # Find matches
    matches = matcher(doc)
    return matches

def find_gender_using_word(doc, token_indexes):
    [ref_start, ref_end] = token_indexes
    
    g_tokens = []
    for token_index in range (ref_start,ref_end):
        
        morph = str(doc[token_index].morph)
    
        if "Masc" in morph:
            g_tokens.append("Masc")
        elif "Fem" in morph:
            g_tokens.append("Fem")
    return calculate_prob_gender(g_tokens)

def find_gender_using_pronoun(doc, token_indexes):

    # Store the pronoun that refers to the noun
    pronoun_reference = []
    for ind in range(token_indexes[0], token_indexes[1]):
        token = doc[ind]
        # Find the head verb of the noun which should be a linking verb if the noun is a complement
        head_verb = token.head
        # Check each child of the head verb to find a nominal subject
        for child in head_verb.children:
            if child.dep_ in ['nsubj', 'nsubjpass'] and child.pos_ == 'PRON':
                pronoun_reference += child.morph.get("Gender")

    label, prob = calculate_prob_gender(pronoun_reference)
    return label, prob

def find_gender_using_coreference(doc, token_indexes, explain = False):
    
    
    token_to_cluster = {}
    cluster_to_token = {}

    for i, chain in enumerate(doc._.coref_chains):
        cluster_to_token[i] = []
        for mention in chain:
            token_to_cluster[mention[0]] = i
            cluster_to_token[i].append(mention[0])
        

    coref_words, coref_gen = [], []

    for token_index in range (token_indexes[0], token_indexes[1]):
        if token_index not in token_to_cluster:
            continue
        cluster_index = token_to_cluster[token_index]
        tokens_in_cluster = cluster_to_token[cluster_index]
        for i in tokens_in_cluster:
            coref_words.append(str(doc[i]))
            gen = doc[i].morph.get("Gender")
            if len(gen) == 0:
                coref_gen.append("")
            elif len(gen) == 1:
                coref_gen.append(gen[0])
            else:
                raise Exception(f"Found word with more genders: {str(doc[i])}, with: {gen}")
        
    label, prob = calculate_prob_gender(coref_gen)

    if explain:
        explanation = "\n".join([f"{w} --> {g}" for w, g in zip(coref_words, coref_gen) if g != ""])
        print (f"'{str(doc[token_index])}' classified as {label}, because was co-referenced with the words:\n{explanation}")
    return label, prob
        
    
    
def calculate_prob_gender(genders):
    fem_count = genders.count("Fem")
    masc_count = genders.count("Masc")

    if len(genders) == 0:
        return "NC", np.nan 
        
    if fem_count > masc_count:
        label = "Fem"
        prob = fem_count/(fem_count + masc_count)
    elif fem_count < masc_count:
        label = "Masc"
        prob = masc_count/(fem_count + masc_count)
    else:
        label = "NC"
        prob = 0.5

    return label, prob
    
