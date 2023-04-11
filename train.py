import argparse
import datetime
import itertools
from math import ceil
import os
from time import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import read_yaml, get_available_device
from models import build_model
from datasets import build_dataloaders


def parse_cmd_args():
    parser = argparse.ArgumentParser(prog='BasilAI trainer')
    parser.add_argument('--config', default=os.path.join('config', 'train.yaml'),
                        help='Path to training config')
    args = parser.parse_args()
    return args


def train(
        device, model, optimizer, train_dataloader, tokenizer,
        val_dataloader=None, epochs=0, iterations=100, logs_path='logs',
        checkpoints_path='checkpoints', val_steps=50, autosave_secs=None,
):
    model.to(device)

    total_steps = iterations
    if epochs:
        total_steps = len(train_dataloader) * epochs
    else:
        epochs = ceil(iterations / len(train_dataloader))
    
    val_iter = None
    if val_dataloader is not None:
        val_iter = itertools.cycle(iter(val_dataloader))

    session_timestamp = str(datetime.datetime.now())
    session_timestamp = session_timestamp.replace(" ", "").replace(":", "-").replace(".", "-")
    logs_path = os.path.join(
        logs_path,
        session_timestamp,
    )
    os.makedirs(logs_path)
    autosave_tstamp = None
    if autosave_secs is not None:
        checkpoints_path = os.path.join(
            checkpoints_path,
            session_timestamp,
        )
        os.makedirs(checkpoints_path)
        autosave_tstamp = time()
        tokenizer.save(os.path.join(checkpoints_path, 'tokenizer'))

    writer = SummaryWriter(log_dir=logs_path)

    step = 0
    with tqdm(total=total_steps) as pbar:
        for epoch in range(epochs):
            for train_batch in train_dataloader:
                model.train()
                optimizer.zero_grad()

                in_tokens = torch.Tensor(train_batch['encoder_input']).long().to(device)
                dec_inputs = torch.Tensor(train_batch['decoder_input']).long().to(device)
                dec_targets = torch.Tensor(train_batch['decoder_output']).long().to(device)

                loss, acc = model(
                    tokens=in_tokens,
                    dec_input=dec_inputs,
                    dec_target=dec_targets,
                )
                
                loss.backward()
                optimizer.step()

                if device != torch.device('cpu'):
                    loss = loss.cpu()
                    acc = acc.cpu()
                loss = loss.detach().item()
                acc = acc.detach().item()
                pbar.set_description(f'Step: {step} Epoch: {epoch} Loss: {loss} Acc: {acc}')
                writer.add_scalar('Loss/Train', loss, step)
                writer.add_scalar('Accuracy/Train', acc, step)
                
                if val_steps and (step % val_steps == 0) and (val_iter is not None):
                    val_batch = next(val_iter)
                    in_tokens = val_batch['encoder_input']
                    dec_inputs = val_batch['decoder_input']
                    dec_targets = val_batch['decoder_output']

                    val_result = ''
                    for sample_idx in range(len(in_tokens)):
                        text_input = tokenizer.decode_line(in_tokens[sample_idx]).replace('PAD', '')
                        target_output = tokenizer.decode_line(dec_targets[sample_idx]).replace('PAD', '')
                        prediction = model.infer(text_input, tokenizer)
                        if len(prediction.replace(' ', '')):
                            val_result += f'Sample # {sample_idx} | {text_input} | {prediction} | ({target_output}) \n'

                    if len(val_result):
                        writer.add_text('Validation/Samples', val_result, global_step=step)

                    model.train()
                    val_loss, val_acc = model(
                        tokens=torch.Tensor(in_tokens).to(model.device),
                        dec_input=torch.Tensor(dec_inputs).to(model.device),
                        dec_target=torch.Tensor(dec_targets).to(model.device),
                    )
                    if device != torch.device('cpu'):
                        val_loss = val_loss.cpu()
                        val_acc = val_acc.cpu()
                    val_loss = val_loss.detach().item()
                    val_acc = val_acc.detach().item()
                    writer.add_scalar('Loss/Val', val_loss, step)
                    writer.add_scalar('Accuracy/Val', val_acc, step)
                
                pbar.update(1)
                step += 1

                if autosave_tstamp is not None:
                    if time() > autosave_tstamp + autosave_secs:
                        autosave_tstamp = time()
                        model_path = os.path.join(checkpoints_path, f"chatter_ep_{epoch}_step_{step}")
                        model.save(model_path)

                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    model_path = os.path.join(checkpoints_path, f"chatter_final_ep_{epoch}_step_{step}")
    model.save(model_path)
    print('Completed')


def run(args):
    training_config = read_yaml(args.config)

    device = training_config['device']
    if device == 'auto':
        device = get_available_device()
    device = torch.device(device)

    model, optimizer = build_model(
        training_config['model'],
        training_config['optimizer'],
    )
    print(model)

    train_dataloader, val_dataloader, tokenizer = \
        build_dataloaders(training_config['dataset'])
    print(f'Train dataset size: {len(train_dataloader)}')
    print(f'Val dataset size: {len(val_dataloader)}')

    train(
        device=device,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
        val_dataloader=val_dataloader,
        epochs=training_config['epochs'],
        iterations=training_config['iterations'],
        logs_path=training_config['logs'],
        checkpoints_path=training_config['checkpoints'],
        val_steps=training_config['validation_steps'],
        autosave_secs=training_config['autosave_mins'] * 60.,
    )


if __name__ == '__main__':
    run(parse_cmd_args())
