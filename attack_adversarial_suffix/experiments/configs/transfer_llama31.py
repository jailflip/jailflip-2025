import os

os.sys.path.append("..")
from experiments.configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.tokenizer_paths = [
        "meta-llama/Llama-3.1-8B-Instruct"  # TODO: specify your model path here
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "meta-llama/Llama-3.1-8B-Instruct"  # TODO: specify your model path here
   ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["llama-3"]
    config.devices = ["cuda:0"]

    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.model_name_str = 'llama31'

    return config