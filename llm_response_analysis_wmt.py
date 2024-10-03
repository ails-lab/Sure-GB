import pandas as pd
import pickle
from utils import extract_occupations_from_resp, Knowledge
from gender import *
from utils import *
from tqdm import tqdm 

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--filename', type=str, required=True, help="Path to the input file")
parser.add_argument('--language', type=str, required=True, help="Language code (e.g., 'en')")
parser.add_argument('--output_filename', type=str, required=True, help="Path to the output file (.pickle)")

args = parser.parse_args()
    
filename = args.filename
language = args.language
output_filename = args.output_filename


responses = pd.read_csv(filename)

el = Greek()
en = English()
fr = French()

# Step 3: Connect with Knowledge
knowledge = Knowledge("../ISCO-08-EN.csv", column = "Definition")


if language == "el":
    check_coreference = False
    nlp = el.nlp
if language == "fr":
    check_coreference = True
    nlp = fr.nlp
if language == "en":
    check_coreference = True
    nlp = en.nlp
            
            
analysis = []
for _, single_resp in tqdm(responses.iterrows()):
    text = single_resp["text"]
    llm_resp = single_resp["llm-response"]
    
    try:
    
        # Step 2: Analyze output of LLM to extract occupations and definitions if are existed
        list_of_occs = extract_occupations_from_resp(llm_resp)

        #### remove hallucinations ####
        extracted_occups = []
        titles = set()
        for v in list_of_occs:
            if len(v["title"].split(" ")) > 3: 
                continue
            if v["title"] in titles:
                continue
            titles.add(v["title"])
            extracted_occups.append(v.copy())

        for v in extracted_occups.copy():
            occ = v["title"]
            if len(link_occupation_mentions_with_lemmatization(nlp, text, occ)) == 0:
                extracted_occups.remove(v)
        #### edn of remove hallucinations ####

        list_of_occs = extracted_occups.copy()

        analysis_single = {
            "text": text, 
            "llm-response": llm_resp,
            "language": language,
            "occupations": []
        }

        for row in list_of_occs:
            index, p = knowledge.connect(row['definition'])[0]

            genders = find_gender(nlp, text, [row], check_coreference = check_coreference)
            for gender in genders:
                analysis_single["occupations"].append({
                        "index": index,
                        "title": row['title'],
                        "p": round (p*100, 2),
                        "kg": knowledge.describe_occ_dict(index),
                        "gender": gender,
                    })

        analysis.append(analysis_single.copy())

        with open (output_filename, "wb") as handle:
            pickle.dump(analysis, handle)
            
    except Exception as e:
        print (e)