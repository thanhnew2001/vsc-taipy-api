import torch
from flask import Flask, jsonify, request

import ctranslate2
import transformers

app = Flask(__name__)

model = "taipy14"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = ctranslate2.Generator(model, device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained(model)

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        text = data.get('inputs', '')
        max_new_tokens = data.get('parameters', {}).get('max_new_tokens', 64)
        mode = data.get('mode', '')

        if mode == 'TaipyMarkdown':
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            results = generator.generate_batch([tokens], max_length=max_new_tokens, include_prompt_in_result=False)
            generated_text = tokenizer.decode(results[0].sequences_ids[0])
            return jsonify({'generated_text': generated_text})

        else:
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            results = generator.generate_batch([tokens], max_length=max_new_tokens, include_prompt_in_result=False)
            generated_text = tokenizer.decode(results[0].sequences_ids[0])
            return jsonify({'generated_text': generated_text})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
