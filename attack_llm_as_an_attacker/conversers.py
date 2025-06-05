import time
import common
from language_models import GPT
import argparse
import torch
from config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MAX_PARALLEL_STREAMS, USER_INPUT_TEMPLATE

def load_attack_and_target_models(args):
    # first determine the device
    attack_device = f"cuda:{args.device_attack}"
    target_device = f"cuda:{args.device_target}"

    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = args.attack_model,
                        max_n_tokens = args.attack_max_n_tokens,
                        max_n_attack_attempts = args.max_n_attack_attempts,
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        device=torch.device(attack_device),
                        args=args
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name = args.target_model,
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        device=torch.device(target_device),
                        args=args
                        )
    return attackLM, targetLM

class AttackLM():
    """
        Base class for attacker language models.

        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self,
                model_name: str,
                max_n_tokens: int,
                max_n_attack_attempts: int,
                temperature: float,
                top_p: float,
                args: argparse.Namespace,
                device:torch.device):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name, device=device)
        self.args = args

        if "vicuna" in model_name or "llama" in model_name:
            self.model.extend_eos_tokens()

    def set_zero_query_time_tracker(self):
        self.args.attack_query_times = 0
        self.args.target_query_times = 0
        self.args.evaluator_query_times = 0

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model.
        Only valid outputs in proper JSON format are returned. If an output isn't generated
        successfully after max_n_attack_attempts, it's returned as None.

        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.

        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """

        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."

        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \""""

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            # Get prompts
            conv.append_message(conv.roles[0], prompt)
            full_prompts.append(conv.to_openai_api_messages())

        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs
            #     Query the attack LLM in batched-queries
            #     with at most MAX_PARALLEL_STREAMS-many queries at a time (TAP implementation, https://github.com/dreadnode/parley)
            outputs_list = []
            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset))

                if right == left:
                    continue
                self.args.attack_query_times += len(full_prompts_subset[left:right])

                outputs_list.extend(
                                    self.model.batched_generate(full_prompts_subset[left:right],
                                                        max_n_tokens = self.max_n_tokens,
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
                )

            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                attack_dict, json_str = common.extract_json(full_output)

                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)

            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate

            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

            time.sleep(2)

        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

class TargetLM():
    """
        Base class for target language models.

        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self,
            model_name: str,
            max_n_tokens: int,
            temperature: float,
            top_p: float,
            device: torch.device,
            args: argparse.Namespace,
            preloaded_model: object = None
            ):

        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.args = args
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name, device=device)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def set_zero_query_time_tracker(self):
        self.args.attack_query_times = 0
        self.args.target_query_times = 0
        self.args.evaluator_query_times = 0

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)

        # ----- Step1, get the full_prompts obj, either via fschat or tokenizer.apply_chat_template() ---- #
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):

            # implement the jailflip target model template...
            prompt += f"\n{USER_INPUT_TEMPLATE.format(question=self.args.goal)}"

            conv.append_message(conv.roles[0], prompt)
            if True:
                # to use the fschat conv.to_openai_api_messages() function
                full_prompts.append(conv.to_openai_api_messages())


        # ------------------------------------- Step2, batch generate ------------------------------------ #
        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))

            if right == left:
                continue
            self.args.target_query_times += len(full_prompts[left:right])
            outputs_list.extend(
                                self.model.batched_generate(full_prompts[left:right],
                                                            max_n_tokens = self.max_n_tokens,
                                                            temperature = self.temperature,
                                                            top_p = self.top_p
                                                        )
            )
        return outputs_list

def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)

    # only considering use api models...
    if model_name in ["claude-3-haiku", "gpt", "gemini", "gemini-1.5"]:
        lm = GPT(model_path)
    else:
        raise ValueError(f"Not supported model {model_name}")

    return lm, template

def get_model_path_and_template(model_name):
    # supported api models listed below
    full_model_dict={
        "claude-3-haiku":{
            "path":"claude-3-haiku-20240307",
            "template":"claude-3-haiku"
        },
        "gpt":{
            "path":"gpt-4o-2024-11-20",
            "template":"gpt"
        },
        "gemini-1.5":{
            "path":"gemini-1.5-flash",
            "template":"gemini-1.5"
        },
        "gemini":{
            "path":"gemini-2.0-flash",
            "template":"gemini"
        }
    }
    assert model_name in full_model_dict.keys()

    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template