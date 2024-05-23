from argparse import ArgumentParser
import pandas as pd
from urllib.request import urlopen
from PIL import Image
import timm
import torch


def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()


def main(args):
    
    cid_to_spid = load_class_mapping(args.class_mapping)
    spid_to_sp = load_species_mapping(args.species_mapping)
        
    device = torch.device(args.device)

    model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=args.pretrained_path)
    model = model.to(device)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    img = None
    if 'https://' in args.image or 'http://' in  args.image:
        img = Image.open(urlopen(args.image))
    elif args.image != None:
        img = Image.open(args.image)
        
    if img != None:
        img = transforms(img).unsqueeze(0)
        img = img.to(device)
        output = model(img)  # unsqueeze single image into batch of 1
        top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
        top5_probabilities = top5_probabilities.cpu().detach().numpy()
        top5_class_indices = top5_class_indices.cpu().detach().numpy()

        for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
            species_id = cid_to_spid[cid]
            species = spid_to_sp[species_id]
            print(species_id, species, proba)



if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--image", type=str, default='https://lab.plantnet.org/LifeCLEF/PlantCLEF2024/single_plant_training_data/PlantCLEF2024singleplanttrainingdata/test/1361687/bd2d3830ac3270218ba82fd24e2290becd01317c.jpg') #Orchis simia
    #bd2d3830ac3270218ba82fd24e2290becd01317c.jpg
    parser.add_argument("--class_mapping", type=str) #'class_mapping.txt'
    parser.add_argument("--species_mapping", type=str) #'species_id_to_name.txt'
    
    parser.add_argument("--pretrained_path", type=str) #model_best.pth.tar

    parser.add_argument("--device", type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)
