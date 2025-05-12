import importlib
import os

from asr.optim.adamp import AdamP
from asr.optim.radam import RAdam
from asr.optim.novograd import Novograd

# automatically import any Python files in the optim/ directory
scheduler_dir = os.path.dirname(__file__)
for file in os.listdir(scheduler_dir):
    if os.path.isdir(os.path.join(scheduler_dir, file)) and file != "__pycache__":
        for subfile in os.listdir(os.path.join(scheduler_dir, file)):
            path = os.path.join(scheduler_dir, file, subfile)
            if subfile.endswith(".py"):
                scheduler_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"asr.optim.scheduler.{scheduler_name}")