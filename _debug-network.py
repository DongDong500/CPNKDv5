if __name__ == "__main__":

    import network
    import argparse
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deeplabv3plus_resnet50', type=str)
    parser.add_argument('--is_rgb', default=True, type=bool)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--output_stride', default=32, type=int)
    parser.add_argument('--separable_conv', default=False, type=bool)
    args = parser.parse_args()
    
    try:    
        if args.model_name.startswith("deeplab"):
            model = network.model.__dict__[args.model_name](channel=3 if args.is_rgb else 1, 
                                                        num_classes=args.num_classes, output_stride=args.output_stride)
            if args.separable_conv and 'plus' in args.model_name:
                network.convert_to_separable_conv(model.classifier)
            utils.set_bn_momentum(model.backbone, momentum=0.01)
        else:
            model = network.model.__dict__[args.model_name](channel=3 if args.is_rgb else 1, 
                                                        num_classes=args.num_classes)
    except:
        raise Exception("<load model> Error occured while loading a model.")
    
    print(utils.summary(model, (3, 256, 256), device='cpu'))
    
    import torch
    import segmentation_models_pytorch as smp
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3plus_resnet50', pretrained=True)
    model = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights='imagenet', classes=2, encoder_output_stride=16, decoder_atrous_rates=(6, 12, 18) )
    print(utils.summary(model, (3, 256, 256), device='cpu'))