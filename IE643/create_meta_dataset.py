import os
import torch
import argparse
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np

def get_query(queries):
    print("\n=====> GENERATING QUERIES <=====\n")
    model = models.resnet18(pretrained=True).to(device)
    layer = model._modules.get('avgpool')

    def copy_embeddings(m, i, o):
        """Copy embeddings from the penultimate layer."""
        o = o[:, :, 0, 0].detach().numpy().tolist()
        outputs.append(o)

    outputs = []
    # attach hook to the penultimate layer
    _ = layer.register_forward_hook(copy_embeddings)

    model.eval()
    
    _ = model(queries)

    list_embeddings = [item for sublist in outputs for item in sublist]

    query_train = list_embeddings
    query_train = torch.tensor(query_train)


    print(f'query_shape_train: {len(query_train)}, each query:{len(query_train[0])}')

    del outputs
    del list_embeddings
    return query_train

def process_dataset(dataset_path, save_path, device):
    # code to process each dataset
    dataset_name = os.path.basename(dataset_path)
    tr_path = os.path.join(dataset_path, 'tr')
    te_path = os.path.join(dataset_path, 'te')

    # Define transformations for resizing images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Check GPU availability
    if device.type == 'cuda':
        print(f" \n Using GPU: {torch.cuda.get_device_name(device.index)} \n")
    else:
        print("\n Using CPU for computations. \n")

    # Process training data
    label_map = {}  # Dictionary to map string labels to integers
    label_counter = 0

    # Process training data
    x_train = []
    y_train = []
    for class_folder in os.listdir(tr_path):
        class_path = os.path.join(tr_path, class_folder)
        label = class_folder  # Assuming folder names are class labels
        if label not in label_map:  # If label not mapped yet, assign an integer
            label_map[label] = label_counter
            label_counter += 1
        label = label_map[label]  # Assign the integer value
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            img = transform(img).to(device)
            x_train.append(img)
            y_train.append(label)

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train, device=device)

    # Process testing data
    x_test = []
    y_test = []
    for class_folder in os.listdir(te_path):
        class_path = os.path.join(te_path, class_folder)
        label = class_folder  # Assuming folder names are class labels
        if label not in label_map:  # If label not mapped yet, assign an integer
            label_map[label] = label_counter
            label_counter += 1
        label = label_map[label]  # Assign the integer value
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            img = transform(img).to(device)
            x_test.append(img)
            y_test.append(label)

    x_test = torch.stack(x_test)
    y_test = torch.tensor(y_test, device=device)

    # Calculate the number of images to take as queries (5% of the total)
    num_queries = int(0.05 * len(x_train))

    # Randomly select a subset of the images
    indices = torch.randperm(len(x_train))[:num_queries]
    queries = x_train[indices]

    # Pass the queries to the get_query method
    query = get_query(queries)



    # Create dictionary
    dataset_dict = {
        'task': dataset_name,
        'clss': list(set(y_train.cpu().numpy())),  # Unique labels
        'nclss': len(set(y_train.cpu().numpy())),  # Number of classes
        'x_train': x_train.cpu() if device.type == 'cuda' else x_train,
        'y_train': y_train.cpu() if device.type == 'cuda' else y_train,
        'x_test': x_test.cpu() if device.type == 'cuda' else x_test,
        'y_test': y_test.cpu() if device.type == 'cuda' else y_test,
        'query': query
    }

    # Save dictionary as .pt file in the specified saving path
    file_path = os.path.join(save_path, f"meta_test_{dataset_name}.pt")
    torch.save(dataset_dict, file_path)
    print(f"\n File meta_test_{dataset_name}.pt  saved successfully.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process datasets and create .pt files')
    parser.add_argument('--dataset_path', type=str, default='path/to/your/data', help='Path to datasets folder')
    parser.add_argument('--save_path', type=str,default='path/for/outcomes', help='Path to save .pt files')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index to use (default: -1 for CPU)')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path

    # Determine device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    if not os.path.isdir(dataset_path):
        print("Invalid path to datasets.")
    else:
        # Create the save_path directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
    
        datasets = os.listdir(dataset_path)
        for dataset in datasets:
            print(f'\n=====>PROCESSING {dataset}<=====\n')
            dataset_folder = os.path.join(dataset_path, dataset)
            if os.path.isdir(dataset_folder):
                process_dataset(dataset_folder, save_path, device)
            else:
                print(f'{dataset} does not exist')
