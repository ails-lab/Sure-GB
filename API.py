from occ_models import HF_Model, find_occupations_aws
from utils import extract_occupations_from_resp, Knowledge
from gender import * 
import os 

import torch
import transformers
import itertools

from flask import Flask, request, jsonify

app = Flask(__name__)



def extract_occupations_and_gender(text, language, knowledge):
    
    global en
    global el
    global fr
    
    
    # Step 1: Find Occupations using LLM
    llm_resp = find_occupations_aws(text)
    # Step 2: Analyze output of LLM to extract occupations and definitions if are existed
    list_of_occs = extract_occupations_from_resp(llm_resp)

    responses = []
    for row in list_of_occs:
        index, p = knowledge.connect(row['definition'])[0]


        if language == "el":
            check_coreference = False
            nlp = el.nlp
        if language == "fr":
            check_coreference = True
            nlp = fr.nlp
        if language == "en":
            check_coreference = True
            nlp = en.nlp

        gender = find_gender(nlp, text, [row], check_coreference = check_coreference)
        
        responses.append({
                "index": index,
                "title": row['title'],
                "p": round (p*100, 2),
                "kg": knowledge.describe_occ_dict(index),
                "gender": gender,
                "text": text,
                "language": language

            })

    return responses

def find_word_start_end(src, token_index):
    words = src.strip().split()
    current_position = 0
    
    for i, word in enumerate(words):
        start_position = current_position
        end_position = start_position + len(word) - 1
        
        if i == token_index:
            return [start_position, end_position + 1]
        
        current_position = end_position + 2  # +1 for the space, +1 because end_position is inclusive


def align(src, tgt, ds, dt, model, tokenizer):
#     sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    sent_src, sent_tgt = [str(s) for s in ds.nlp(src)], [str(s) for s in dt.nlp(tgt)]
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
        sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )

    al = {}
    for i, j in sorted(align_words):
        al[i] = [j, sent_src[i], sent_tgt[j]]

    return al

def search_g(resp_target, m):
    for row2 in resp_target:
        for r2 in row2["gender"]:
            s2, e2 = r2["tokens"]
            g2 = r2["gender"]
            if s2 == m:
                return row2["title"], g2



def api_func(text_source, text_target, language_source, language_target):
    global knowledge
    global model
    global tokenizer
    
    
    response = ""
    resp_source = extract_occupations_and_gender(text_source, language_source, knowledge)
    resp_target = extract_occupations_and_gender(text_target, language_target, knowledge)
    
    
    response = f"Analysis of source text:\n\n {str(resp_source)}\n\nAnalysis of target text: \n\n {str(resp_target)}\n\nResults:\n"
    mapping = align(text_source, text_target, en, el, model, tokenizer)

    for row1 in resp_source:
        for r1 in row1["gender"]:
            s1, e1 = r1["tokens"]
            g1 = r1["gender"]

            t2, g2 = search_g(resp_target, mapping[s1][0])

            if g1 != g2:
                response += f"    - Found gender shift in the word: {row1['title']} --> {t2}, from {g1} to {g2}\n\n"
   
                   
    return response

@app.route('/analyze', methods=['POST'])
def api_call():
    
    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')
    lang1 = data.get('lang1')
    lang2 = data.get('lang2')
    
    allowed_languages = ['en', 'el', 'fr']

    # Validate that all inputs are provided
    if not all([text1, text2, lang1, lang2]):
        return jsonify({'error': 'text1, text2, lang1, and lang2 are required.'}), 400

    # Validate that the languages are among the allowed ones
    if lang1 not in allowed_languages or lang2 not in allowed_languages:
        return jsonify({'error': f'Languages must be one of {allowed_languages}.'}), 400
    
    result = api_func(text1, text2, lang1, lang2)
    return jsonify({'result': result})

    
    
if __name__ == "__main__":
    
    global knowledge
    global en
    global el
    global fr
    
    model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    knowledge = Knowledge("ISCO-08-EN.csv", column = "Definition")
    
    el = Greek()
    en = English()
    fr = French()

    os.environ['aws_access_key_id'] = '' # add aws secret access key id
    os.environ['aws_secret_access_key'] = '' # add aws secret access key
    
    app.run(host='0.0.0.0', port=8090)

