import time
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets.aircraft import Aircraft

#import models.mobilenetv2_ad_s
#import models.mobilenetv2_ad_s_flops
import models.mobilenetv2_ra
#from torchsummary import summary

from fvcore.nn import FlopCountAnalysis

from torchinfo import summary

def loadWeightFromBaseline(base_model, new_model):
    
    base_model_sdict_items = list(base_model.state_dict().items())
    new_model_sdict = new_model.state_dict()

    index = 0

    print(type(base_model_sdict_items))

    for new_param in new_model_sdict.items():

        new_param_name = new_param[0]

        if "residual_adapter" not in new_param_name:
            base_param_name = base_model_sdict_items[index][0]
            base_param = base_model_sdict_items[index][1]

            new_model_sdict[new_param_name] = base_param
            #print(new_param_name)

            print("base: {},   new: {}".format(base_param_name, new_param_name))
            index += 1

    new_model.load_state_dict(new_model_sdict)
    


if __name__ == "__main__":

    preprocess = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_augment = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    batch = 256

    #imagenet_val = torchvision.datasets.ImageFolder("/home/hangyeol/datasets/ImageNet/train", transform=preprocess)
    aircraft_dataset = Aircraft("/mnt/ssd/aircraft", train="trainval", transform=preprocess, download=False)
    test_dataset = Aircraft("/mnt/ssd/aircraft", train="test", transform=preprocess, download=False)

    
    loader = DataLoader(aircraft_dataset, batch_size=batch,  shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch,  shuffle=True, num_workers=4)


    dataset_size = len(aircraft_dataset)
    test_dataset_size = len(test_dataset)
    print(dataset_size)
    print(test_dataset_size)
    
    model = models.mobilenetv2_ra.mobilenet_v2_ra(pretrained=False)
    base_model = torchvision.models.mobilenet_v2(pretrained=True)

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    load_model_path = "./baseline_pretrained.pth"

    #base_model.load_state_dict(torch.load(load_model_path))
    loadWeightFromBaseline(base_model, model)


    for param in model.parameters():
        param.requires_grad = False


    for name, param in model.named_parameters():
        param.requires_grad = False
        if "residual_adapter" in name:
            print(name)
            param.requires_grad = True
    
    num_classes = len(aircraft_dataset.classes)
    print(num_classes)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    torch.nn.init.xavier_uniform_(model.classifier[1].weight)

    model.to(device)
    base_model.to(device)

    #summary(model, input_size=(3, 224, 224))
    #summary(base_model, input_size=(3, 224, 224))
    summary(model, input_size=(1, 3, 224, 224))


    # model.eval()

    model_save_path = "./saved_models/finetune_s2.pth"
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=0.0005, eps=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    print_step = 1000
    total_epoch = 100
    decay_start_epoch = 40


    for epoch in range(total_epoch):
        print_loss = 0.0
        start_time = time.time()
        corrects = 0

        count = 0
        total_loss = 0.0 
        test_count = 0
        test_corrects = 0

        #Train Process

        model.train()
        if epoch + 1 >= decay_start_epoch:
            scheduler.step()


        current_lr = optimizer.param_groups[0]['lr']

        print("epoch {0}".format(epoch + 1))
        print("learning rate: {}".format(current_lr))
        for inputs, labels in loader:
            # start_time_step = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)

            #flops = FlopCountAnalysis(model, inputs)
            #print(flops.total())
            
            # print(inputs.type())

            
            # with torch.no_grad():
            #high.start_counters([events.PAPI_FP_OPS,])
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            #x=high.stop_counters()

            #print("flops for a epoch: {}".format(x))

            _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)

            count += 1
            print_loss += loss.item()
            total_loss += loss.item()
            # if count == 10:
            #    break

            # final_time_step = time.time() - start_time_step

            # print("{0} / {1}   {2} secs".format(batch * count, dataset_size, final_time_step))

            if count % print_step == 0:
                print("{0} / {1} loss: {2}".format(batch * count, dataset_size, print_loss / print_step))
                print_loss = 0.0

            # print("step loss: {0}".format(loss.item()))
            


        # epoch_acc = corrects.double() / dataset_size
        epoch_acc = corrects.double() / (batch * count)

        
        print("train total accuracy = {0}, total_loss = {1}".format(epoch_acc, total_loss / count))
        model.eval()
        base_model.eval()
        # Test Process
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            
            # print(inputs.type())

            # optimizer.zero_grad()

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

        
            test_corrects += torch.sum(preds == labels.data)

            test_count += 1
            #if test_count == 10:
            #    break

            #print("{0} / {1}".format(batch * test_count, dataset_size))


    
        epoch_acc = test_corrects.double() / test_dataset_size
        # epoch_acc = corrects.double() / (batch * test_count)

        final_time_epoch = (time.time() - start_time) / 60

        print("time elapased: {0} mins".format(final_time_epoch))
        
        print("test total accuracy = {0}".format(epoch_acc))
        
                

    torch.save(model.state_dict(), model_save_path)
    
    print("saved model")
