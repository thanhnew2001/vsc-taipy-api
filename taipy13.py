import os
import threading
import csv
import pandas as pd

from flask import Flask
from pyngrok import ngrok
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub

from flask import request, jsonify

model_name = "taipy12-ct2"
model = GeneratorCT2fromHfHub(
        # load in int8 on CUDA
        model_name_or_path=model_name,
        device="cuda",
        compute_type="int8_float16",
        # tokenizer=AutoTokenizer.from_pretrained("{ORG}/{NAME}")
)

def generate_text_batch(prompt_texts, max_length=64):
    outputs = model.generate(prompt_texts, max_length=max_length, include_prompt_in_result=False)
    return outputs

app = Flask(__name__)
port = "5000"

def read_csv_as_string(file_path):
    print(file_path)
    with open(file_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        csv_content = "\n".join(",".join(row) for row in reader)
    return csv_content

# Prepare few shots learning
def prompt_localllm_fsl_plot(user_input):
    context = read_csv_as_string("context_data.csv")
    print(context)
    full_prompt = f"""{context} 
        \n{user_input};<"""
    return full_prompt


# Define Flask routes
@app.route("/")
def index():
    return "Hello from Colab!"

@app.route("/api/generate", methods=["POST"])
def generate_code():
    try:
        # Get the JSON data from the request body
        data = request.get_json()
        
        # Extract 'inputs' and 'parameters' from the JSON data
        inputs = data.get('inputs', '')
        parameters = data.get('parameters', {})
        # Extract the 'max_new_tokens' parameter
        max_new_tokens = parameters.get('max_new_tokens', 64)
        mode = data.get('mode')
        if mode=='TaipyMarkdown':
        # Get only the most recent instruction line
            lines = inputs.strip().splitlines()
            last_line = lines[-1] 
            full_prompt = "In Taipy, "+ prompt_localllm_fsl_plot(last_line)
            generated_text = "<"+generate_text_batch([full_prompt], max_new_tokens)[0]
        else:
        # Get whole text as usual 
        # Call the generate_text_batch function with inputs and max_new_tokens
            generated_text = generate_text_batch([inputs], max_new_tokens)[0]

        return jsonify({
        "generated_text": generated_text,
        "status": 200
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# Start the Flask server in a new thread
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
