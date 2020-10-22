import argparse




def args_input():
    
    
    parser=argparse.ArgumentParser()
    
    #train
    parser.add_argument('--save_dir', type=str, default='save', help='path for checkpoint saving')
    parser.add_argument('--flower_dir', type=str, default='flowers',help='path for flowers')
    parser.add_argument('--arc', type=str, default='vgg19',help='path for network architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--hidden_units', type=int, default= 1024 , help='hidden layers')
    parser.add_argument('--epochs', type=int, default=1, help='number of epocs')
    parser.add_argument('--gpu', type=bool, default=False,  help=' requesting GPU')
    
    #predict
    parser.add_argument('--top_k', type=int, default=5, help='the top K most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', type=str, action='store',help='names of categories')
    parser.add_argument('--image_path', default='flowers', nargs='?', action="store", help='procces this image with predict')
    parser.add_argument('--checkpoint', default='save', nargs='?', action="store", help='path to saved checkpoint')
    
    in_arg=parser.parse_args()
    
    return in_arg