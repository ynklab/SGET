# implementation based on Alpaca-LoRA https://github.com/tloen/alpaca-lora/blob/main/finetune.py
# this file is modifed for SGET. (c) 2024 Ryoma Kumon
import os
import sys
from typing import List
import argparse
import torch
import transformers
from datasets import load_dataset

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os.path as osp
from typing import Union



class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

    def get_batch_response(self, output):
        responses = []
        for i in range(len(output)):
            decode_strs = output[i].split(self.template["response_split"])
            decode_str = ""
            if len(decode_strs) > 1:
                decode_str = decode_strs[1]
            else:
                decode_str = output[i].split(self.template["response_split2"])[1]
            decode_str = decode_str.split("<unk>")[0]
            decode_str = decode_str.replace("</s>", "")
            responses.append(decode_str.strip())
        return responses
def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_path: str = "yahma/alpaca-cleaned",
    dev_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve

    prompt_template_name: str = "cogs",  # The prompt template to use, will default to alpaca.
    seed: int = 0,
):
    torch.manual_seed(seed)
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"seed: {seed}\n"
            f"base_model: {base_model}\n"
            f"train_path: {train_path}\n"
            f"dev_path: {dev_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()


        # print(len(result["input_ids"]))

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if train_path.endswith(".json"):
        train_data = load_dataset("json", data_files=train_path)["train"]
    else:
        train_data = load_dataset(train_path)
    if dev_path.endswith(".json"):
        val_data = load_dataset("json", data_files=dev_path)["train"]
    else:
        val_data = load_dataset(dev_path)
    

    # if resume_from_checkpoint:
    #     # Check the available weights and load them
    #     checkpoint_name = os.path.join(
    #         resume_from_checkpoint, "pytorch_model.bin"
    #     )  # Full checkpoint
    #     if not os.path.exists(checkpoint_name):
    #         checkpoint_name = os.path.join(
    #             resume_from_checkpoint, "adapter_model.bin"
    #         )  # only LoRA model - LoRA config above has to fit
    #         resume_from_checkpoint = (
    #             False  # So the trainer won't try loading its state
    #         )
    #     # The two files above have a different name depending on how they were saved, but are actually the same.
    #     if os.path.exists(checkpoint_name):
    #         print(f"Restarting from {checkpoint_name}")
    #         adapters_weights = torch.load(checkpoint_name)
    #         model = set_peft_model_state_dict(model, adapters_weights)
    #     else:
    #         print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    print(train_data[0])
    if val_set_size > 0:
        # train_val = data["train"].train_test_split(
        #     test_size=val_set_size, shuffle=True, seed=42
        # )
        train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
        val_data = val_data.shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = train_data.shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    
    print(train_data[0])
    print(torch.cuda.device_count())
    print(os.environ.get("LOCAL_RANK", 1))
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    with torch.autocast("cuda"): 
        trainer.train()

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="",
        help="The base model to use for training. Required.",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="yahma/alpaca-cleaned",
        help="The path to the training data.",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="yahma/alpaca-cleaned",
        help="The path to the development data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora-alpaca",
        help="The directory to save the trained model to.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size to use for training.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=32,
        help="The micro batch size to use for training.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--cutoff_len",
        type=int,
        default=256,
        help="The cutoff length to use for training.",
    )
    parser.add_argument(
        "--val_set_size",
        type=int,
        default=0,
        help="The size of the validation set to use for training.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="The r parameter to use for LoRA.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha parameter to use for LoRA.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout rate to use for LoRA.",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        type=str,
        default=["q_proj", "v_proj"],
        help="The target modules to use for LoRA.",
    )
    parser.add_argument(
        "--train_on_inputs",
        type=bool,
        default=False,
        help="Whether to train on inputs.",
    )
    parser.add_argument(
        "--group_by_length",
        type=bool,
        default=False,
        help="Whether to group by length.",
    )
    parser.add_argument(
        "--prompt_template_name",
        type=str,
        default="cogs",
        help="The name of the prompt template to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The seed to use for training.",
    )
    args = parser.parse_args()
    train(**vars(args))