from tqdm.auto import tqdm
import torch

def evaluate(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for (images, labels) in testloader:
            if torch.cuda.is_available():
                images = images.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
    model.train()
    return 100. * correct / total

def train(model, dataloader, testloader, optimizer, criterion, epochs=10, writer=None, scheduler=None):
    running_loss = 0.0
    model.train()
    for epoch in tqdm(range(epochs)):
        if scheduler:
            scheduler.step()
        running_loss = 0
        for i, data in enumerate(dataloader):
            # data = (inputs, targets, teacher_scores(optional))
            if torch.cuda.is_available():
                data = tuple([x.cuda() for x in data])

            optimizer.zero_grad()
            outputs = model(data[0].float())
            
            loss = criterion(outputs, *data[1:])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
                    
            
        acc = evaluate(model, testloader)
        print("Epoch {} accuracy = {:.2f}%".format(epoch + 1, acc))

        if writer:
            writer.add_scalar('accuracy', acc, epoch)
            writer.add_scalar('training loss', running_loss/len(dataloader), epoch)
        running_loss = 0.0
    model.eval()