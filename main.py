import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora


class Args:
    pass

def get_custom_arguments():
    args = Args()


    args.seed = 1
    # Dataset arguments
    args.root_path = "data"

    #args.dataset = "oxford_flowers"
    args.dataset = "hm"
    args.shots = 1 #4
    # Model arguments
    args.backbone = 'ViT-B/16'
    # Training arguments
    args.lr = 2e-4
    args.n_iters = 2
    args.batch_size =  1 #32
    # LoRA arguments
    args.position = 'mid'
    args.encoder = "both"
    args.params = 'q'
    args.r = 2
    args.alpha = 1
    args.dropout_rate = 0.25

    args.save_path = None
    args.filename = 'lora_weights'
    args.eval_only = False

    # CPU or CUDA
    args.core = "CPU"

    return args


def main():

    # Load config file
    #args = get_arguments()
    args = get_custom_arguments()
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
        
    dataset = build_dataset(args.dataset, args.root_path, args.shots, preprocess)
    
    if args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        
    train_loader = None
    if not args.eval_only:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_transform, is_train=True, shuffle=True, num_workers=8)

    run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)

if __name__ == '__main__':
    main()
    print ("------- Done ------------------")