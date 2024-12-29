# Author    : zhiping
import torch


def train(train_loader, model, criterion, optimizer):
    model.train()
    model = model.cuda()
    criterion = criterion.cuda()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def vail(test_loader, model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    print(f"accurate on test set: {100 * correct / total}, {correct}, {total}")
