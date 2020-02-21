

import argparse

# Create Parse using ArgumentParser


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default = 'flowers/',help ='path to the folder of pet images',type = str)
    parser.add_argument('--device',default = 'cuda',help = 'choose the device GPU',type = str)
    parser.add_argument('--save_dir',default = 'checkpoint',help = 'location to save',type =str)
    parser.add_argument('--arch',default = 'vgg16',help = 'if argmt is mentioned, uses Vgg13 else Alexnet will be used',type = str )
    parser.add_argument('--learning_rate',default = '0.001',help = 'learning rate with default value 0.001',type = float)
    parser.add_argument('--epochs',default = 4,help = 'list the number of epochs',type = int)
    parser.add_argument('--path_to_image', type = str, default = '28/image_05230.jpg',help = 'image path to test')
    parser.add_argument('--checkpoint_dir', type = str, default = 'image_checkpoint.pth', help = 'path to save checkpoints')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',help = 'json file with category names')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Top K classes')
    return parser.parse_args()

 