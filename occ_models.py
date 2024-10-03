from utils import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
import boto3
import json
from botocore.exceptions import ClientError

class HF_Model:
    
    
    def __init__(self, hf_path = "mistralai/Mistral-7B-v0.1"):
        self.hf_path = hf_path
        self.tokenizer, self.model = load_model(self.hf_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        
    def find_occupations(self, text, use_prompt = True):
        
        if use_prompt:
            prompt = f"""You are a helpful AI assistant. This is a conversation between you and a User. You is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision.
User:
In the following text, identify the occupation titles that are explicitly stated and provide the occupation title along with a brief definition in the following format:

Occupation title: [Occupation title as it is referred to in the text]
Definition: [Definition]
If no occupation is identified, please respond with:
"No occupation found."

User:
{text}

You:"""
        else:
            prompt = text

        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens = 150, temperature=1)

        output = self.tokenizer.batch_decode(generated_ids)[0]
        output = output.split("\nYou:")[1]
        return output
        
        
hf_path = "meta-llama/Meta-Llama-3.1-405B"
tokenizer = AutoTokenizer.from_pretrained(hf_path)

def find_occupations_aws(text, tokenizer = tokenizer, aws_model_id= "meta.llama3-1-405b-instruct-v1:0"):
    
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name="us-west-2"
    )
        
    system_prompt = """You are a helpful AI assistant. This is a conversation between you and a User. You is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision."""
    
    user_instr = f"""In the following text, identify the occupation titles that are explicitly stated and provide the occupation title along with a brief definition in the following format:

Occupation title: [Occupation title exactly as it is referred to in the text]
Definition: [Definition]
If no occupation is identified, please respond with:
"No occupation found."

Text:
{text}
"""
    
    chat = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_instr},
    ]
    formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    
    # Format the request payload using the model's native structure.
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 500,
        "temperature": 1,
        "top_p": 0.9
    }


    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=aws_model_id, body=request)

        input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
        output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])

        # print(f"Input tokens: {input_tokens}")
        # print(f"Output tokens: {output_tokens}")


    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{aws_model_id}'. Reason: {e}")
        exit(1)


    # Decode the response body.
    model_response = json.loads(response["body"].read())


    # Extract and print the response text.
    response_text = model_response["generation"]
    return response_text
        
