import torch
import numpy as np
import random
import os
import torchvision.models as models
from tqdm import tqdm
import time
import argparse
from zoo_loader import *





class ModelZoo:
    def __init__(self, model_zoo_path, m_train_path, device, seed, batch_size, dataset_path, load_model_zoo_path, load_m_train_path):
        self.seed = seed
        self.models = {}
        self.load_model_zoo_path = load_model_zoo_path
        self.load_m_train_path = load_m_train_path

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.model_zoo_path = model_zoo_path
        self.m_train_path = m_train_path         
        self.noise = None
        self.device = device

        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Check if model_zoo file exists, load if yes
        if os.path.isfile(self.load_model_zoo_path):
            self.load_model_zoo()
        else:
            self.zoo = {'dataset': [], 'topol': [], 'f_emb': [], 'acc': [], 'n_params': []}


        # Check if m_train file exists, load if yes
        if os.path.isfile(self.load_m_train_path):
            self.load_m_train()
        else:
            self.trainpt_dict = {}


    def load_model_zoo(self):
        # Load model zoo from model_zoo.pt
        print('\n=====> LOADING model_zoo.pt <=====\n')
        self.zoo = torch.load(self.load_model_zoo_path)


    def load_m_train(self):
        # Load training instances from m_train.pt
        print('\n=====> LOADING m_train.pt <=====\n')
        self.trainpt_dict = torch.load(self.load_m_train_path)
        
    
    def init_loaders(self):
        # Get loaders for train, test, and validation data
        print('\n=====> LOADING dataset and loaders for train, test and validation <=====\n')
               
        self.tr_dataset = ZooDatasets(mode='train', batch_size=self.batch_size, data_path=self.dataset_path)
        self.te_dataset = ZooDatasets(mode='test', batch_size=self.batch_size, data_path=self.dataset_path)
        self.val_dataset = ZooDatasets(mode='validation', batch_size=self.batch_size, data_path=self.dataset_path)
        
        
        
        
        
        
    def create_zoo(self, noise_path, learn, patience, epochs, n_nets):     
        self.noise = torch.load(noise_path)
        self.init_loaders()

            
            
        for query_dataset in self.tr_dataset.get_dataset_list():
            
            
            
            print('quesry_dataset',query_dataset)
            
            self.tr_dataset.set_dataset(query_dataset)
            self.te_dataset.set_dataset(query_dataset)
            self.val_dataset.set_dataset(query_dataset)

            tr_loader = self.tr_dataset.get_loader(mode='train')
            te_loader = self.te_dataset.get_loader(mode='test')
            val_loader = self.val_dataset.get_loader(mode='validation')
            nclass = self.tr_dataset.get_nclss()
            print(f"\nDataset : {query_dataset}, nclss:{nclass}\n")
            
            
            


            for i in range(n_nets):
                # Get neural network model and topology information
                print(f'\n=====> Generating {i+1}th network  <=====\n')
                topol, net = self.get_net(nclass)

                # Training the model and obtaining accuracy
                lss = torch.nn.CrossEntropyLoss()
                optim = torch.optim.SGD(net.parameters(), lr=learn, momentum=0.9, weight_decay=4e-5)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim , float(epochs))
                
                acc = self.train( net, optim, scheduler, lss, query_dataset, tr_loader, val_loader,  nclass, topol, learn)  

                # Calculating number of parameters
                n_params = self.n_param(net)
                f_emb = self.f_emb(net)

                
                del net
                del optim
                del lss
                
                self.zoo['dataset'].append(query_dataset)
                self.zoo['topol'].append(topol)
                self.zoo['acc'].append(acc)
                self.zoo['f_emb'].append(f_emb)
                self.zoo['n_params'].append(n_params)
                
                print(self.zoo)
                
            x_query_train, x_query_test = self.get_query(tr_loader, te_loader)
            print(f'x_query:{len(x_query_train)}, x_query-test :{len(x_query_test)}')
            clss = self.tr_dataset.get_clss()
            self.save_trainpt(query_dataset, clss, nclass, x_query_test, x_query_train)
        self.save_zoo( zoo_dict = self.zoo)
                
        
        
                
                


    def get_net(self, nclss):
        print("\n=====>Generating sub_net<=====\n")
        super_net_name = "ofa_supernet_mbv3_w12"

        # Load the super network from the 'mit-han-lab/once-for-all' repository
        super_net = torch.hub.load('mit-han-lab/once-for-all', super_net_name, pretrained=True).eval()

        # Sample an active subnet configuration from the super network
        sampled_config = super_net.sample_active_subnet()

        # Extract topology information from the sampled configuration
        pre_topol = list(sampled_config.values())
        topol = [item for sublist in pre_topol for item in sublist]

        # Split the topology into kernel sizes, expansion ratios, and depths
        ks = topol[:20] 
        e = topol[20:40]
        d = topol[40:]

        # Set the active subnet in the super network using the sampled topology
        super_net.set_active_subnet(ks=ks, e=e, d=d)
        active_subnet = super_net.get_active_subnet(preserve_weight=True)
        active_subnet.classifier = torch.nn.Linear(1536, nclss)
        active_subnet = active_subnet.to(self.device)

        return topol, active_subnet

    


        

    def f_emb(self, net):
        print("\n=====>Generating f_emb<=====\n")
        model = net
        model.to(self.device)
        layer = model.feature_mix_layer
        
        def copy_embeddings(module, input, output):
            # Assuming the output shape is (batch_size, channels, height, width)
            embeddings = output.squeeze()  # Remove dimensions of size 1
            outputs.append(embeddings.cpu().detach().numpy())


        outputs = []
        _ = layer.register_forward_hook(copy_embeddings)

        model.eval()
        noise = self.noise.to(self.device)
        _ = model(noise)
        f_emb = outputs[-1]
        f_emb = torch.tensor(f_emb)
        del outputs
        print(f_emb.shape)
        return f_emb
    
    
    

    def train(self, model, optim, scheduler, lss, dataset, train_loader, val_loader,  nclss, topol, learn):
        print("\n=====> Training started <=====\n")
        #self.model = model
        print(f'Starting Training for:{dataset} with model topology:{topol}')
        lr = learn
        counter = 0
        best_val_loss = 1000000
        val_acc = []
        train_acc = []

        
        
        
        for ep in range(epochs):
            curr_ep = ep
            ep_loss_tr = 0.0
            ep_loss_val = 0.0
            ep_tr_time = 0
            total_tr = 0
            correct_tr = 0
            st = time.time()
            
            
            model.train()
            for b_id, batch in tqdm(enumerate(train_loader)):
                optim.zero_grad()
                
                x,y = batch
                output = model(x.to(self.device))
                loss = lss(output, y.to(self.device))
                pred = torch.argmax(output, dim = 1)
                total_tr += y.size(0)
                correct_tr += (pred == y.to(self.device)).sum().item()

                loss.backward()
                optim.step()
                scheduler.step()
                        
                tr_loss = loss.item()
                        
                ep_loss_tr += tr_loss * x.size(0)
                
            ep_loss_tr = ep_loss_tr/len(train_loader)
            acc_tr = (100*correct_tr)/total_tr
            train_acc.append(acc_tr)


                        
            model.eval()
            total_val = 0
            correct_val = 0

                        
                        
            for v_id, (x,y) in tqdm(enumerate(val_loader)):
                outputs = model(x.to(self.device))
                loss_v = lss(outputs, y.to(self.device))

                       
                val_loss = loss_v.item()
                
                ep_loss_val += val_loss * x.size(0)
                        
                pred_v = torch.argmax(outputs, dim = 1)
                total_val += y.size(0)
                correct_val += (pred_v == y.to(self.device)).sum().item() 
                        
            acc = (100*correct_val)/total_val
            val_acc.append(acc)
            ep_loss_val = ep_loss_val/len(val_loader)
            dura = time.time() -st 
            
            print('\nEpoch: {}/{}, Train Loss: {:.8f} , Val Loss: {:.8f}, Train Accuracy:{:.8f},  Val Accuracy: {:.8f}, Epoch Time: {:.8f}'.format(ep + 1, epochs, ep_loss_tr, ep_loss_val, acc_tr ,acc, dura))
            
            
    
            if ep_loss_val < best_val_loss:
                best_val_loss = ep_loss_val
                counter = 0
                
            else:
                counter += 1
                
            if counter >= patience:
                print(f"Early stopping on, {ep}th, epoch")
                break
               
            #print(f'ep : {ep}, loss:{ep_loss_tr}, val_loss:{ep_loss_val}, acc:{acc}, time:{dura}')


        return acc
            
            
                        
                        
                        
                        
    
    def n_param(self, model):
        # Calculate the number of parameters in the model
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    

    def get_query(self, train_loader, test_loader):
        print("\n=====> GENERATING QUERIES <=====\n")
        model = models.resnet18(pretrained=True)
        layer = model._modules.get('avgpool')

        def copy_embeddings(m, i, o):
            """Copy embeddings from the penultimate layer."""
            o = o[:, :, 0, 0].detach().numpy().tolist()
            outputs.append(o)

        outputs = []
        # attach hook to the penultimate layer
        _ = layer.register_forward_hook(copy_embeddings)

        model.eval()
        X, y = next(iter(train_loader))
        _ = model(X)

        list_embeddings = [item for sublist in outputs for item in sublist]

        query_train = [torch.tensor(a) for a in list_embeddings]

        print(f'query_shape_train: {len(query_train)}, each tensor:{query_train[0].shape}')

        outputs = []
        list_embeddings = []

        model.eval()

        X, y = next(iter(test_loader))
        _ = model(X)

        list_embeddings = [item for sublist in outputs for item in sublist]

        query_test = [torch.tensor(a) for a in list_embeddings]

        print(f'query_shape_test: {len(query_test)}, each tensor:{query_test[0].shape}')
        del outputs
        del list_embeddings
        return query_train, query_test
    
    
    
    

    def save_zoo(self, zoo_dict):
        # Save model zoo to model_zoo.pt
        print(f'\n=====>SAVING {zoo_dict} in model_zoo.pt <=====\n')

        file = os.path.join(self.model_zoo_path, 'model_zoo.pt')
        torch.save( zoo_dict , file)

        
        
    def save_trainpt(self, dataset, clss, nclss, x_test, x_train) :
        # Save training instances to m_train.pt
        print(f'\n=====>SAVING {dataset} in m_train.pt <=====\n')
        temp = {}
        temp['task'] = dataset
        temp['clss'] = clss
        temp['nclss'] = nclss
        temp['x_query_test'] = x_test
        temp['x_query_train'] = x_train
        
        self.trainpt_dict[dataset] = temp
        
        fpath = os.path.join(self.m_train_path, 'meta_train.pt') 
        torch.save(self.trainpt_dict, fpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Zoo Construction')
    parser.add_argument('--noise_path', type=str, default='/kaggle/input/noise-tans/noise.pt', help='Path to noise file')
    parser.add_argument('--learn', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch_size to be used for training')
    parser.add_argument('--patience', type=int, default=5, help='Patience value')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training generated networks')
    parser.add_argument('--dataset_path', type=str, default='/kaggle/input/model-zoo-tans/v14/geon/final_data/MetaTrain', help='Path to datasets file')
    parser.add_argument('--model_zoo_path', type=str, default='kaggle/working/', help='Path to save model zoo file')
    parser.add_argument('--m_train_path', type=str, default='kaggle/working/', help='Path to save meta train file')
    parser.add_argument('--n_nets', type=int, default= 10, help='number of networks to be generated per dataset')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not, if 1 then use GPU, else use CPU')
    parser.add_argument('--seed', type=int, default=777, help='Random seed value')
    parser.add_argument('--load_model_zoo_path', type=str, default='kaggle/input/model_zoo/model_zoo.pt', help='Path for saved model zoo file')
    parser.add_argument('--load_m_train_path', type=str, default='kaggle/input/meta_train/meta_train.pt', help='Path for saved meta train file')


    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f" \n Using GPU: {torch.cuda.get_device_name(device.index)} \n")
    else:
        device = torch.device('cpu')
        print("\n Using CPU for computations. \n")

    noise_path = args.noise_path
    learn = args.learn
    patience = args.patience
    epochs = args.epochs
    model_zoo_path = args.model_zoo_path
    m_train_path = args.m_train_path
    n_nets = args.n_nets
    seed = args.seed
    batch_size = args.batch_size
    dataset_path = args.dataset_path
    load_model_zoo_path = args.load_model_zoo_path
    load_m_train_path = args.load_m_train_path

    model_zoo = ModelZoo(model_zoo_path, m_train_path, device, seed, batch_size, dataset_path, load_model_zoo_path, load_m_train_path)
    model_zoo.create_zoo(noise_path, learn, patience, epochs, n_nets)




