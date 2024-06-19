from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from lora import Prompter
import json, sys
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_lora(model_name, lora_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True,torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.float16)
    return model, tokenizer

def translate_line(model, tokenizer, line):
    prompter = Prompter('cogs')
    prompt = prompter.generate_prompt(line['instruction'], line['input'])
    encoding = tokenizer(prompt, return_tensors="pt").to('cuda')
    output_ids = model.generate(**encoding, max_new_tokens=256)
    pred = tokenizer.decode(output_ids[0])
    pred = prompter.get_response(pred)
    print(pred)
    return pred

def translate_batch(model, tokenizer, batch):
    prompter = Prompter('cogs')
    batch_size = len(batch["instruction"])
    prompts = [prompter.generate_prompt(batch["instruction"][i],
                                        batch["input"][i])
                                        for i in range(batch_size)]
    encodings = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
    output_ids = model.generate(**encodings, max_new_tokens=256)
    pred = tokenizer.batch_decode(output_ids)
    print(pred)
    pred = prompter.get_batch_response(pred)
    return pred

def translate_file(path, model_name, lora_path, save_path):
    model, tokenizer = load_lora(model_name, lora_path)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    tokenizer.padding_side = "left"
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    model.bfloat16().eval()
    data = load_dataset('json', data_files=path)['train']
    data_loader = DataLoader(data, batch_size=32)

    preds = []
    for batch in data_loader:
        pred = translate_batch(model, tokenizer, batch)
        preds += pred

    # for line in tqdm(data):
    #     pred = translate_line(model, tokenizer, line)
    #     preds.append(pred)


    results = []
    for line, output_text in zip(data, preds):
        result_line = output_text
        results.append(result_line)

    with open(save_path, 'w') as f:
        f.write('\n'.join(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--lora_path', type=str, default='lora')
    parser.add_argument('--save_path', type=str, default='results.txt')
    parser.add_argument('--data_path', type=str, default='data.json')
    args = parser.parse_args()
    translate_file(args.data_path, args.base_model, args.lora_path, args.save_path)

    
