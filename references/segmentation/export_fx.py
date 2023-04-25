import torch
import torchvision

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['fcn_resnet50', 'fcn_resnet101'])
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    args = parser.parse_args()
    paths = {
        "voc": 21,
        "voc_aug": 21,
        "coco": 21,
    }
    num_classes = paths[args.dataset]
    if args.model == 'fcn_resnet50':
        model = torchvision.models.segmentation.fcn_resnet50(weights="FCN_ResNet50_Weights.DEFAULT",
                                                            num_classes=num_classes,
                                                            aux_loss = args.aux_loss
                                                            )
    elif args.model == 'fcn_resnet101':
        model = torchvision.models.segmentation.fcn_resnet101(weights="FCN_ResNet50_Weights.DEFAULT",
                                                            num_classes=num_classes,
                                                            aux_loss = args.aux_loss
                                                            )
    graph = torch.fx.Tracer().trace(model)
    traced = torch.fx.GraphModule(model, graph)
    torch.save(traced, f'{args.model}_fx.pt')