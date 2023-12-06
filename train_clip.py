import time
import random
import torch
import CLIP
import argparse
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from code.misc import LABEL_1, LABEL_2, DATA_ROOT
from code.dataset import TaobaoDataset, MyTaobaoDataset
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# tune the model
def train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25):
    since = time.time()
    for epoch in range(num_epochs):
        model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0

        for i, (inputs, labels) in tqdm(enumerate(dataloaders['train'])):
            inputs = inputs.to(device)
            nonzero_indices = torch.nonzero(labels != 0, as_tuple=False)[:, 1].int()
            filter_ = nonzero_indices != len(LABEL_1)
            inputs, labels, nonzero_indices = inputs[filter_, ...], labels[filter_, ...], nonzero_indices[filter_].tolist()
            texts = [LABEL_2[label] for label in nonzero_indices]
            texts = CLIP.tokenize(texts).to(device)
            #labels = labels.to(device)
            ground_truth = torch.arange(inputs.shape[0]).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            logits_per_image, logits_per_text = model(inputs, texts)

            def isnan_inf(t):
                return t.isnan().any() or t.isinf().any()
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,ground_truth)) / 2 
            # backward
            total_loss.backward()
            
            # update
            optimizer.step()

            # print loss statistics
            running_loss += total_loss.item()
            if i % 100 == 0:
                print(f'step: {i}: loss: {total_loss.item()}')
            # save to tensorboard
            writer.add_scalar('loss/training', total_loss.item(), epoch * len(dataloaders['train']) + i)
        
        # print epoch statistics
        epoch_loss = running_loss / len(dataloaders['train'])
        print('Epoch {} Loss (training): {:.4f}'.format(epoch, epoch_loss))

        # save model
        torch.save(model.state_dict(), './model/clip_res50_epoch_{}.pth'.format(epoch))
        # save the latest model name to txt file
        with open('./model/last_model_clip_res50.txt', 'w') as f:
            f.write('clip_res50_epoch_{}.pth'.format(epoch))

        # validate
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            correct = 0.1
            total = 0.1
            for inputs, labels in dataloaders['train']:
                # randomly pass 90% to save time.
                if random.random() > 0.1:
                    continue
                inputs = inputs.to(device)
                labels = labels.to(device)
                nonzero_indices = torch.nonzero(labels != 0, as_tuple=False)[:, 1].int()
                filter_ = nonzero_indices != len(LABEL_1)
                inputs, labels, nonzero_indices = inputs[filter_, ...], labels[filter_, ...], nonzero_indices[filter_].tolist()
                texts = [LABEL_2[label] for label in nonzero_indices]
                texts = CLIP.tokenize(texts).to(device)
                ground_truth = torch.arange(inputs.shape[0]).to(device)

                logits_per_image, logits_per_text = model(inputs, texts)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,ground_truth)) / 2 

                running_loss += total_loss.item()
                probs = logits_per_image.softmax(dim=-1)
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == ground_truth).sum().item()

            epoch_loss = running_loss / len(dataloaders['train'])
            epoch_acc = correct / total
            print('Epoch {} Loss (train): {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            writer.add_scalar('loss/train', epoch_loss, epoch)
            writer.add_scalar('accuracy/train', epoch_acc, epoch)

            running_loss = 0.0
            correct = 0.1
            total = 0.1
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                nonzero_indices = torch.nonzero(labels != 0, as_tuple=False)[:, 1].int()
                filter_ = nonzero_indices != len(LABEL_1)
                inputs, labels, nonzero_indices = inputs[filter_, ...], labels[filter_, ...], nonzero_indices[filter_].tolist()
                texts = [LABEL_2[label] for label in nonzero_indices]
                texts = CLIP.tokenize(texts).to(device)
                ground_truth = torch.arange(inputs.shape[0]).to(device)

                logits_per_image, logits_per_text = model(inputs, texts)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,ground_truth)) / 2 

                running_loss += total_loss.item()
                probs = logits_per_image.softmax(dim=-1)

                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == ground_truth).sum().item()

            epoch_loss = running_loss / len(dataloaders['val'])
            epoch_acc = correct / total
            print('Epoch {} Loss (val): {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
            writer.add_scalar('loss/val', epoch_loss, epoch)
            writer.add_scalar('accuracy/val', epoch_acc, epoch)

def main(num_classes=len(LABEL_1)+1, tune_full_model=False, batch_size=32, num_epochs=25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = CLIP.load("RN50", device=device)
    model = model.float()
    text = CLIP.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # if not tune_full_model:
    #     # freeze all layers
    #     for param in model.parameters():
    #         param.requires_grad = False 
    # print("Params to learn:")
    # for name,param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print("\t",name)
    train_loader = torch.utils.data.DataLoader(
        dataset=MyTaobaoDataset(train=True, data_root=DATA_ROOT, preprocess=preprocess),
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=MyTaobaoDataset(train=False, data_root=DATA_ROOT, preprocess=preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=True
    )
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    writer = SummaryWriter('/root/tf-logs/')
    train_model(model, dataloaders, criterion, optimizer, writer, num_epochs=25)


if __name__ == '__main__':
    # pass args
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=len(LABEL_1)+1)
    parser.add_argument('--tune_full_model', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    main(args.num_classes, args.tune_full_model, args.batch_size)