from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
from search import Search
from googletrans import Translator


def load_model(model_name, device_map = {"": "cuda:1"}, load_in_4bit = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map=device_map,
                                                 load_in_4bit=True
                                                 )
    return tokenizer, model

# def generate(chat, model, tokenizer, max_new_tokens = 1000, num_return_sequences = 1):
#     """
#     This function takes as an input the logs of a chat along with a model and a tokenizer and generates a list of repsonses (or response) of the LM
#     """
#     input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to("cuda")
#     outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, do_sample=False) #top_p=0.92, top_k=500) 
        
#     return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# def decode_chat_output(output):
#     return output.split("[/INST]")[-1].strip()

# def generate_thoughts(chat, model, tokenizer, num_of_thoughts = 5):
#     """
#     Sample num_of_thoughts different responses 
#     """
#     # outputs = generate(chat, model, tokenizer, max_new_tokens = 250, num_return_sequences = 3)
#     responses = []
#     for i in range (num_of_thoughts):
#         output = generate(chat, model, tokenizer)
#         responses.append(decode_chat_output(output))
#     return responses    


# class Chat:
#     def __init__(self, model, tokenizer, system_message):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.chat = [{"role": "system", "content": system_message}]

#     def add_users_input(self, users_input):
#         self.chat.append({"role": "user", "content": users_input})

#     def answer(self, max_new_tokens = 250):
#         output = generate(self.chat, self.model, self.tokenizer, max_new_tokens=max_new_tokens)[0]
#         output = decode_chat_output(output)
#         self.chat.append({"role": "assistant", "content": output})
#         return output


# def create_occupation_promp(text):
#     prompt = f"""In the following text identify the occupations and provide the occupation title along with a small definition of this occupation, in the following format: 
# Occupation title: [Occupation title as it is referred in the text]
# Definition: [Definition]

# {text}
#  """
# #     prompt = f"""This is a conversation between User and a friendly chatbot. This chatbot is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.

# # User:
# # In the following text, identify if there are occupation titles explicitly stated. If there are not respond with:

# # Zero occupations are explicitly stated in this text.

# # Otherwise provide the occupation title along with a brief definition in the following format:

# # Occupation title: [Occupation title as it is referred to in the text]
# # Definition: [Definition]


# # Input:
# # {text}
# # """
# #     prompt = f"""In the following text, identify the occupation titles that are explicitly stated and provide the occupation title along with a brief definition in the following format:

# # Occupation title: [Occupation title as it is referred to in the text]
# # Definition: [Definition]
# # If no occupation is identified, please respond with:
# # "No occupation found."

# # Here are some examples: 

# # Input: 
# # The doctor consulted with the nurse to determine the best treatment plan for the patient.

# # Occupation title: Doctor
# # Definition: A licensed medical professional who diagnoses and treats illnesses and injuries.

# # Occupation title: Nurse
# # Definition: A healthcare professional who cares for patients, administers medication, and supports doctors in the provision of healthcare services.

# # Input: 
# # They worked together to ensure the best possible care for their patient.

# # No occupation found.

# # Input:
# # {text}

# """
    return prompt

def extract_occupations_from_resp(text):
    """
    Extracts occupation titles and descriptions from text using regular expressions.
    
    Args:
      text: The text to extract occupations from.
    
    Returns:
      A list of dictionaries, where each dictionary represents an occupation with keys
      'title' and 'definition'.
    """
    text = text.lower()
    occupations = []
    pattern = r"occupation title: (.*?)\ndefinition:(.*?)(?=\noccupation title:|$)"
#     pattern = r"occupation title: (.*?)\ndefinition:(.*?)(?=\n\d+\. occupation title:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for title, description in matches:
        title = re.sub(r'\([^)]*\)', '', title).strip()
        occupations.append({
          "title": title.strip(),
          "definition": description.strip()
        })
    return occupations

def extract_occupations_from_resp_el(text, translator):
    """
    Extracts occupation titles and descriptions from text using regular expressions.
    
    Args:
      text: The text to extract occupations from.
    
    Returns:
      A list of dictionaries, where each dictionary represents an occupation with keys
      'title' and 'definition'.
    """
    text = text.lower()
    occupations = []
    pattern = r"τίτλος επαγγέλματος: (.*?)\nορισμός:(.*?)(?=\nΤίτλος επαγγέλματος:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    
    for title, description in matches:
        title = re.sub(r'\([^)]*\)', '', title).strip()
        occupations.append({
          "title": title.strip(),
          "definition": translator.translate(description.strip(), "el", "en")
        })
    return occupations



class GoogleTranslate:
    
    def __init__(self):
        self.translator = Translator(service_urls=['translate.googleapis.com'])
        
    def translate(self, source_text, source_lang, target_lang):
        return self.translator.translate(source_text, src= source_lang, dest=target_lang).text
    
    def translate_batch(self, source_texts, source_lang, target_lang):
        return [self.translator.translate(source_text, src= source_lang, dest=target_lang).text for source_text in source_texts]
    


class Knowledge:

    def __init__(self, filename, column):
        self.occ_defs = pd.read_csv(filename, sep =";", dtype={'ISCO 08 Code': str})
        texts = self.occ_defs[column].tolist()
        self.s = Search(texts)

    def connect(self, descr, top_k = 5):
        return self.s.search(descr)[:top_k]

    def describe_occ(self, index):
        row = self.occ_defs.iloc[index]
    
        return f"""Level: {row['Level']}
ISCO 08 Code: {row['ISCO 08 Code']}
Title EN: {row['Title EN']}
Definition: {row['Definition']}
"""
    def describe_occ_dict(self, index):
        row = self.occ_defs.iloc[index]
    
        return {
            "level": row['Level'],
            "title EN": row['Title EN'],
            "definition": row['Definition'],
            "ISCO 08 Code": row['ISCO 08 Code']
        }
