import time
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from datasets.aircraft import Aircraft
from datasets.dtd import DTD
from datasets.flowers import Flowers
from datasets.stanford_dogs_data import dogs
from datasets.cubmini import CUBMini


from torchinfo import summary

from fvcore.nn import FlopCountAnalysis

import pytorch_dataset.cub2011
import pytorch_dataset.cars

def loadWeightFromBaseline(base_model, new_model):
    
    base_model_sdict_items = list(base_model.state_dict().items())
    new_model_sdict = new_model.state_dict()

    index = 0

    print(type(base_model_sdict_items))

    for new_param in new_model_sdict.items():

        new_param_name = new_param[0]
        base_param_name = base_model_sdict_items[index][0]

        if "scale_param" not in new_param_name and "bias_param" not in new_param_name and "point" not in new_param_name and "classifier" not in new_param_name and "res_adap" not in new_param_name and "adapter" not in new_param_name:
                
            #while "15.conv.2" in base_param_name or "15.conv.3" in base_param_name or "16.conv.0" in base_param_name:
            #if index >= len(base_model_sdict_items):
            #    break
            
            #print("{} skipped".format(base_param_name))
            #index+=1
            #base_param_name = base_model_sdict_items[index][0]
            
            #base_param_name = base_model_sdict_items[index][0]
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
        #transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    batch = 64
    does_it_work = 0

    #imagenet_val = torchvision.datasets.ImageFolder("/home/hangyeol/datasets/ImageNet/train", transform=preprocess)
    #aircraft_dataset = DTD("/mnt/ssd/dtd", train="trainval1", transform=preprocess_augment, download=False)
    #aircraft_dataset = dogs("./datasets/dogs", train=True, transform=preprocess_augment, download=True)
    #test_dataset = dogs("./datasets/dogs", train=False, transform=preprocess, download=True)
    #test_dataset = DTD("/mnt/ssd/dtd", train="test1", transform=preprocess, download=False)
    #aircraft_dataset = pytorch_dataset.cub2011.Cub2011("/mnt/ssd/CUB", train=True, transform=preprocess_augment, download=False)
    #test_dataset = pytorch_dataset.cub2011.Cub2011("/mnt/ssd/CUB", train=False, transform=preprocess, download=False)
    #aircraft_dataset = torchvision.datasets.UCF101(root="/mnt/ssd/ucf101/UCF-101", annotation_path="/mnt/ssd/ucf101/annotation", frames_per_clip=1, step_between_clips=10, train=True, transform=preprocess_augment)
    #test_dataset = torchvision.datasets.UCF101(root="/mnt/ssd/ucf101/UCF-101", annotation_path="/mnt/ssd/ucf101/annotation", frames_per_clip=1, step_between_clips=10, train=False, transform=preprocess)

    
    #aircraft_dataset = torchvision.datasets.ImageFolder("/mnt/ssd/ucf_pics/train", transform=preprocess_augment)
    #test_dataset = torchvision.datasets.ImageFolder("/mnt/ssd/ucf_pics/test", transform=preprocess)
    #aircraft_dataset = DTD("/mnt/ssd/dtd", train="trainval1", transform=preprocess_augment, download=False)
    
    

    #aircraft_dataset = pytorch_dataset.cars.Cars("/mnt/ssd/stan_cars", train=True, transform=preprocess_augment, download=False)
    #test_dataset = pytorch_dataset.cars.Cars("/mnt/ssd/stan_cars", train=False, transform=preprocess, download=False)
    aircraft_dataset = Aircraft("/mnt/ssd/aircraft", train="trainval", transform=preprocess_augment, download=True)
    test_dataset = Aircraft("/mnt/ssd/aircraft", train="test", transform=preprocess, download=True)

    #aircraft_dataset = Flowers("/mnt/ssd/flowers", train="trainval", transform=preprocess_augment, download=True)
    #test_dataset = Flowers("/mnt/ssd/flowers", train="test", transform=preprocess, download=True)


    #aircraft_dataset = CUBMini("/mnt/ssd/CUB_small", train="train", transform=preprocess_augment, download=False)
    #test_dataset = CUBMini("/mnt/ssd/CUB_small", train="test", transform=preprocess, download=False)
    
    loader = DataLoader(aircraft_dataset, batch_size=batch,  shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch,  shuffle=False, num_workers=4)


    dataset_size = len(aircraft_dataset)
    test_dataset_size = len(test_dataset)
    print(dataset_size)
    print(test_dataset_size)
    
    #model = models.mobilenetv2_ad_s.mobilenet_v2_ad_s(pretrained=False)
    #model = models.mobilenetv2_ad_ff.mobilenet_v2_ad_ff(pretrained=False)
    #model = models.mobilenetv2_ad_pr.mobilenet_v2_ad_pr(pretrained=False)

    model = torchvision.models.mobilenet_v2(pretrained=False)

    step_size = len(aircraft_dataset) / batch
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    load_model_path = "./saved_models/GammaBetu.pth"

    #base_model.load_state_dict(torch.load(load_model_path))
    #loadWeightFromBaseline(base_model, model)
    #model.load_state_dict(torch.load(load_model_path))

    num_classes = len(aircraft_dataset.classes)
    #um_classes = aircraft_dataset.num_classes
    print("classes: {}".format(num_classes))
    #num_classes = 196
    #model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    #torch.nn.init.xavier_uniform_(model.classifier[1].weight)
 
    
    for param in model.parameters():
        param.requires_grad = False
    
    pc = 1
    for name, param in model.named_parameters():
        #print(name)
        #print("yo")
        #if ("scale_param" in name) or ("bias_param" in name) or ("point" in  name):
        #if "post" in name:
       
        '''
        if "classifier" in name:
        
            print(name)
            param.requires_grad = True
        '''
        
        '''
        if "features.7.conv.0.1.weight" in name or "features.7.conv.0.1.bias" in name:
        
            print(name)
            param.requires_grad = True
        '''
     
        
        if "classifier" in name or "features.0.1" in name or "conv.0.1" in name or "conv.1.1" in name or "conv.3" in name or "features.1.conv.2" in name or "18.1" in name:
            print("grad true: {}".format(name))
            param.requires_grad = True
        else:

            print(name)

        
    #num_classes = len(aircraft_dataset.classes)
    
    print(num_classes)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    torch.nn.init.xavier_uniform_(model.classifier[1].weight)
    
    #model.load_state_dict(torch.load(load_model_path))
    print("loaded models")

    #base_model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    model.to(device)
    #base_model.to(device)


    
    summary(model, input_size=(1, 3, 224, 224))
    #summary(base_model, input_size=(3, 224, 224))

    #model.copyAdap()

    # model.eval()

    model_save_path = "./saved_models/GammaBetra.pth"
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005, eps=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    print_step = 10000000
    total_epoch = 120
    flearning_start_epoch = 25
    decay_start_epoch = 60
    batch_learn_start = 9999

    best_test_acc = 0.0
    count = 0

    is_flops = True
    for epoch in range(total_epoch):
        print_loss = 0.0
        start_time = time.time()
        corrects = 0

        batch_count = 0
        total_loss = 0.0 
        test_count = 0
        test_corrects = 0
        count = 0

        forward_time_avg = 0.0
        backward_time_avg = 0.0

        #Train Process

        model.train()

        

        
        if epoch + 1 >= decay_start_epoch:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print("epoch {0}".format(epoch + 1))
        print("learning rate: {}".format(current_lr))

        load_time_start = time.time()
        
        batch_count = 0
        
        #print("WTH1")
        for inputs, labels in loader:
            #load_time_epoch = (time.time() - load_time_start) / 60
            #print("step")
            #print("load time elapased: {0} mins".format(load_time_epoch))
            #print("WTH2")
            
            #print(batch_count)
            start_time_step = time.time()


            #print("shape")
            #print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            #print(inputs.shape)
            
            if is_flops:
                flops = FlopCountAnalysis(model, inputs)
                back, forward = flops.total()
                print(forward)

                print(back)

                is_flops = False
            

            #print(flops.by_module())
            
            # print(inputs.type())

            
            # with torch.no_grad():
            #high.start_counters([events.PAPI_FP_OPS,])
            
            fstart_time = time.time()
            
            
            outputs = model(inputs)
            
        
            #print("-----------------------batch {}-------------------------".format(batch_count))
            #print("rv------------")
            #print(model.features[2].conv[0][1].running_var.shape)
            #print("weight---------")
            #print(model.features[2].conv[0][1].weight)
            #print("bias---------")
            #print(model.features[2].conv[0][1].bias)
            
            batch_count+=1

            loss = criterion(outputs, labels)

            forward_time_epoch = (time.time() - fstart_time)
            forward_time_avg += forward_time_epoch
       
            #print("forward time elapased: {0} mins".format(forward_time_epoch))

            
            
            bstart_time = time.time()
            
            
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            loss.backward()
            
            #print(prof.key_averages().table())
            #print(prof.key_averages().table())

            
            backward_time_epoch = (time.time() - bstart_time)
            backward_time_avg += backward_time_epoch

            #print("backward time elapased: {0} secs".format(backward_time_epoch))

            

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
            #step_time_epoch = (time.time() - start_time_step) / 60
       
            #print("step time elapased: {0} mins".format(step_time_epoch))

            load_time_start = time.time()

            #scheduler.step()
            #print("scheduler update")

        train_time_epoch = (time.time() - start_time)
       
        print("train time elapased: {0} secs".format(train_time_epoch))

        print("forward time elapased: {0} mins".format(forward_time_avg / step_size))

            
               
        print("backward time elapased: {0} mins".format(backward_time_avg / step_size))


        # epoch_acc = corrects.double() / dataset_size
        epoch_acc = corrects.double() / (batch * count)
        
                
        print("train total accuracy = {0}, total_loss = {1}".format(epoch_acc, total_loss / count))
        model.eval()
        # Test Process
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            
            #flops = FlopCountAnalysis(model, inputs)
            #print("test flops: {}".format(flops.total()))

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


    
        epoch_test_acc = test_corrects.double() / test_dataset_size
        # epoch_acc = corrects.double() / (batch * test_count)

        if best_test_acc < epoch_test_acc:
            print("best acc updated")
            best_test_acc = epoch_test_acc
            torch.save(model.state_dict(), model_save_path)
    
        print("saved model")


        final_time_epoch = (time.time() - start_time)

       

        print("time elapased: {0} secs".format(final_time_epoch))
        
        print("test total accuracy = {0}".format(epoch_test_acc))

        print("best test accuracy = {0}".format(best_test_acc))

        
                

    #torch.save(model.state_dict(), model_save_path)
    
    #print("saved model")
