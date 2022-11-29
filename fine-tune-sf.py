import time
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datasets.aircraft import Aircraft

import models.mobilenetv2_ad_thin_sf
import models.mobilenetv2_org_thin_sf
import models.mobilenetv2_ad_thin_dense


#from torchsummary import summary
from torchinfo import summary

from fvcore.nn import FlopCountAnalysis

def loadWeightFromBaseline(base_model, new_model):
    
    base_model_sdict_items = list(base_model.state_dict().items())
    new_model_sdict = new_model.state_dict()

    index = 0

    print(type(base_model_sdict_items))

    keyword_list = ["point", "thin", "classifier"]
    #keyword_list = ["15.conv.2", "15.conv.3", "16.conv.0"] 
    base_keyword_list = ["15.conv.2", "15.conv.3", "16.conv.0"]


    for new_param in new_model_sdict.items():

        new_param_name = new_param[0]
        base_param_name = base_model_sdict_items[index][0]
        
        #if "point" not in new_param_name or "thin" not in new_param_name and "classifier" not in new_param_name:
        if not any(keyword in new_param_name for keyword in keyword_list):
            #base_param_name = base_model_sdict_items[index][0]
            
            while "15.conv.2" in base_param_name or "15.conv.3" in base_param_name or "16.conv.0" in base_param_name:
                #while any(base_keyword in new_param_name for base_keyword in base_keyword_list):

                if index >= len(base_model_sdict_items):
                    break
                
                print("{} skipped".format(base_param_name))
                index+=1
                base_param_name = base_model_sdict_items[index][0]
            
            if index >= len(base_model_sdict_items):
                break

            base_param = base_model_sdict_items[index][1]
            
            print("base: {},   new: {}".format(base_param_name, new_param_name))

            new_model_sdict[new_param_name] = base_param
            #print(new_param_name)

            index += 1

    new_model.load_state_dict(new_model_sdict)
    


if __name__ == "__main__":

    #preprocessing for Test
    preprocess = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    #preprocessing for Training
    preprocess_augment = torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    batch = 256

    aircraft_dataset = Aircraft("/mnt/ssd/aircraft", train=True, transform=preprocess_augment, download=False)
    test_dataset = Aircraft("/mnt/ssd/aircraft", train=False, transform=preprocess, download=False)

    
    loader = DataLoader(aircraft_dataset, batch_size=batch,  shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch,  shuffle=False, num_workers=4)

    dataset_size = len(aircraft_dataset)
    test_dataset_size = len(test_dataset)
    print(dataset_size)
    print(test_dataset_size)
    print(len(aircraft_dataset.classes))
    
    model = models.mobilenetv2_ad_thin_sf.mobilenet_v2_ad_thin_sf(pretrained=False)
    base_model = torchvision.models.mobilenet_v2(pretrained=True)

    step_size = len(aircraft_dataset) / batch
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    load_model_path = "./baseline_pretrained.pth"

    print("knocking on hells door")
    print(model.feature_list[0][1].printRunningMean())

    
    loadWeightFromBaseline(base_model, model)
    
    print(model.feature_list[0][1].printRunningMean())

    for param in model.parameters():
        param.requires_grad = False

    ae_keyword_list = ["point"]
    #finetune_keyword_list =["middle", "0.1", "post.0", "classifier", "inverted_residuals"]
    #finetune_keyword_list =["classifier", "inverted_residuals"]
    #finetune_keyword_list =["classifier"]
    finetune_keyword_list =["thin", "classifier", "inverted_residuals"]
    #finetune_keyword_list = ["thin"]


    
    for name, param in model.named_parameters():
        print(name)
        if any(keyword in name for keyword in ae_keyword_list):

            print("grad true: {}".format(name))
            param.requires_grad = True
    
    num_classes = len(aircraft_dataset.classes)
    print(num_classes)

    # Replace classifier (last layer) with new classfier
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    torch.nn.init.xavier_uniform_(model.classifier[1].weight)

    #base_model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    # Use GPU for model training
    model.to(device)
    
    # Print information
    #summary(model, input_size=(3, 224, 224))

    model_save_path = "./saved_models/etri_test.pth"

    
    
    # (Thin Adapter Training) Loss Function
    criterion = nn.CrossEntropyLoss()

    # (Autoencoder Training) Loss Function
    rec_criterion = nn.MSELoss()

    # Optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005, eps=0.00001)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    
    # Autoencoder Training Epoch
    ae_epoch = 0
    ae_decay_start_epoch = 100

    # Thin Adapter Training Epoch
    thin_epoch = 100
    thin_decay_start_epoch = 50

    # Best Test Accuracy Value
    best_test_acc = 0.0
    
    #model.setTrainAE(True)
    print("start ae learning")
    for epoch in range(ae_epoch):
        print_loss = 0.0
        start_time = time.time()
        corrects = 0

        count = 0
        total_loss = 0.0 
        test_count = 0
        test_corrects = 0

        forward_time_avg = 0.0
        backward_time_avg = 0.0

        #Train Process

        model.train()
        if epoch + 1 >= ae_decay_start_epoch:
            scheduler.step()

        
        current_lr = optimizer.param_groups[0]['lr']

        print("epoch {0}".format(epoch + 1))
        print("learning rate: {}".format(current_lr))

        load_time_start = time.time()

        for inputs, labels in loader:

            start_time_step = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            fstart_time = time.time()
            
            
            output_list, ae_input_list = model(inputs)

            for index, output in enumerate(output_list):
                
                loss = rec_criterion(output, ae_input_list[index])

                forward_time_epoch = (time.time() - fstart_time)
                forward_time_avg += forward_time_epoch
                
                loss.backward()
            

            optimizer.step()


            optimizer.zero_grad()

            count += 1
            print_loss += loss.item()
            total_loss += loss.item()

            load_time_start = time.time()

        

        print(total_loss)
        print("total_loss = {0}".format(total_loss / count))
    
    model.setTrainAE(False)
    
    for param in model.parameters():

        param.requires_grad = False

    
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in finetune_keyword_list):
            if "out_point" not in name:
                print("grad true: {}".format(name))
                param.requires_grad = True
    
    for name, param in model.named_parameters():
        if param.requires_grad == True:

            print("grad true success: {}    {}".format(name, param.shape))

    
    
    #summary(model, input_size=(3, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005, eps=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    
    for name, module in model.named_modules():
        if "after" in name:
            print(name)
            print(module)
            del module[0]
            del module[0]
            del module[0]
            del module
    
    '''
    del model.feature_list[3].out_point_after_conv
    del model.feature_list[6].out_point_after_conv

    del model.feature_list[10].out_point_after_conv
    del model.feature_list[13].out_point_after_conv
    del model.feature_list[14].inverted_residuals_list[1].out_point_after_conv
    '''


    #print(model.feature_list[3].out_point_after_conv[1])

    #del model.feature_list[3].out_point_after_conv[2]

    summary(model, input_size=(1, 3, 224, 224))

    print("asdffdafadsf")
    print(model.feature_list[0][1])
    print(model.feature_list[0][1].weight.shape)

    print(model.feature_list[0][1].bias.shape)
    print("-------------------round 2-------------------")

    '''
    for name, module in model.named_modules():
        if "after" in name:
            print(name)
            print(module)
    '''

    print("fine tune starts")

    show_flops = True
    
    for epoch in range(thin_epoch):
        print_loss = 0.0
        start_time = time.time()
        corrects = 0

        count = 0
        total_loss = 0.0 
        test_count = 0
        test_corrects = 0

        forward_time_avg = 0.0
        backward_time_avg = 0.0

        #Train Process

        model.train()
        if epoch + 1 >= thin_decay_start_epoch:
            scheduler.step()

        
        current_lr = optimizer.param_groups[0]['lr']

        print("epoch {0}".format(epoch + 1))
        print("learning rate: {}".format(current_lr))

        load_time_start = time.time()

        

        for inputs, labels in loader:
            #load_time_epoch = (time.time() - load_time_start) / 60
       
            #print("load time elapased: {0} mins".format(load_time_epoch))


            start_time_step = time.time()

            inputs = inputs.to(device)
            labels = labels.to(device)

            if show_flops:
            
                flops = FlopCountAnalysis(model, inputs)
                back, forward = flops.total()
                print(forward)
                
                print(back)

                show_flops = False

            


            fstart_time = time.time()
            
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #print(model.feature_list[0][1].printRunningMean())


            forward_time_epoch = (time.time() - fstart_time)
            forward_time_avg += forward_time_epoch 
            #print("forward time elapased: {0} mins".format(forward_time_epoch))

            
            
            bstart_time = time.time()
            loss.backward()
            
            backward_time_epoch = (time.time() - bstart_time)
            backward_time_avg += backward_time_epoch
       

            optimizer.step()

            
            #print("backward time elapased: {0} mins".format(backward_time_epoch))


            optimizer.zero_grad()

            _, preds = torch.max(outputs, 1)

            corrects += torch.sum(preds == labels.data)

            count += 1
            print_loss += loss.item()
            total_loss += loss.item()

            load_time_start = time.time()


        #model.update_temperature()

        train_time_epoch = (time.time() - start_time)
       
        print("train time elapased: {0} secs".format(train_time_epoch))

        print("forward time elapased: {0} mins".format(forward_time_avg / step_size))
            
               
        print("backward time elapased: {0} mins".format(backward_time_avg / step_size))


        epoch_acc = corrects.double() / (batch * count)
        
        print("train total accuracy = {0}, total_loss = {1}".format(epoch_acc, total_loss / count))
        model.eval()
        # Test Process
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                #print(model.feature_list[0][1].printRunningMean())

            test_corrects += torch.sum(preds == labels.data)

            test_count += 1

    
        epoch_test_acc = test_corrects.double() / test_dataset_size

        if best_test_acc < epoch_test_acc:
            print("best acc updated")
            best_test_acc = epoch_test_acc
            torch.save(model.state_dict(), model_save_path)
    
            print("saved model")

        final_time_epoch = (time.time() - start_time)

        print("time elapased: {0} secs".format(final_time_epoch))
        
        print("test total accuracy = {0}".format(epoch_test_acc))

        print("best test accuracy = {0}".format(best_test_acc))

                
