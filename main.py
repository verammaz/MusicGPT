import wandb
import os
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from miditok import REMI, TokenizerConfig
from symusic import Score
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training

from src.models.gpt import GPT
from src.train.train import Trainer
from src.utils.general_utils import set_seed, setup_logging, CfgNode as CN


wandb.login()

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out'

    # pipeline
    C.pipeline = CN()
    C.pipeline.train_gpt = True
    C.pipeline.evaluate = True
    C.pipeline.sample = True

    # data
    C.data = None

    # model
    C.model = GPT.get_default_config()

    # trainer
    C.gpt_trainer = Trainer.get_default_config()

    return C


# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])

    if config.data is None:
        print("no data path provided")

    set_seed(config.system.seed)

    config.model.name = f'{config.model.model_type}_{config.model.n_layer}_{config.model.n_query_head}_{config.model.n_kv_head}'

    if config.model.rope : config.model.name += '_rope'

    #tokenizer = MidiTokenizer(config.data_path, config.tokenizer_path, train=True)
    tokenizer_config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
    tokenizer = REMI(tokenizer_config)

    # construct the model
    config.model.vocab_size = len(tokenizer)
    config.model.block_size = 1024
    
    print(config)
    model = GPT(config.model)
    
    if config.model.pretrained != None:
        pretrained_state_dict = torch.load(config.model.pretrained, weights_only=True)
        model.load_state_dict(pretrained_state_dict, strict=False) 
    
    #setup_logging(config)
    out_dir = os.path.join(config.system.work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)

    wandb.init(project="MusicGen", config=config)

    if config.pipeline.train_gpt:

        midis = list(Path(config.data).resolve().glob("**/*.mid"))

        # Create a Dataset, a DataLoader and a collator to train a model
        dataset = DatasetMIDI(
        files_paths=midis,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"])
        
        collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
        dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)

        # construct the trainer object
        trainer = Trainer(config.gpt_trainer, model, dataloader)

        # iteration callback
        def batch_end_trainer_callback(trainer):
            
            wandb.log({"n_examples" : trainer.n_examples, "train_loss": trainer.loss})
            
            ckpt_path = os.path.join(out_dir, f'{config.model.name}.pt')

            if (trainer.n_iter + 1) % 200 == 0:
                model.eval()
                with torch.no_grad():
                    if config.gpt_trainer.sample: 
                       pass

                torch.save(model.state_dict(), ckpt_path)
            
                # revert model to training mode
                model.train()

        trainer.set_callback('on_batch_end', batch_end_trainer_callback)

        # run the optimization
        trainer.run()

    # evaluate
    if config.pipeline.evaluate:
        pass
    

    