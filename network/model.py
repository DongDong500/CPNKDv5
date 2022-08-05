import segmentation_models_pytorch as SMP

def deeplabv3plus_resnet50(encoder_name, encoder_depth, encoder_weights, encoder_output_stride, 
                            decoder_channels, decoder_atrous_rates, in_channels, classes, 
                            activation, upsampling, aux_params, **kwargs):
    """Constructs a DeepLabV3+ model with a ResNet-50 backbone.

    Args:
        in_channels (int):  A number of input channels for the model, default is 3 (RGB images)
        classes (int):      A number of classes for output mask 
                            (or you can think as a number of channels of output mask)
        encoder_name (str): Name of the classification model that will be used as an encoder (a.k.a
                            backbone) to extract features of different spatial resolution
        encoder_depth (int): A number of stages used in encoder in range [3, 5].
                            Each stage generate features two times smaller in spatial dimentions than previous one 
                            (e.g. for depth 0 we will have features with shapes [(N, C, H, W),], 
                            for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on). (Default is 5)
        encoder_weights (str): One of None (random initialization), “imagenet” (pre-training on ImageNet) 
                            and other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride (int): Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates (tuple): Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels (int): A number of convolution filters in ASPP module. Default is 256
        activation (str):   An activation function to apply after the final convolution layer. Avaliable
                            options are “sigmoid”, “softmax”, “logsoftmax”, “identity”, callable and None. 
                            (Default is None)
        upsampling (int):   Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params (dict):  Dictionary with parameters of the auxiliary output (classification head).
                            Auxiliary output is build on top of encoder if aux_params is not None (default). Supported
                                params:
                                - classes (int): A number of classes
                                - pooling (str): One of “max”, “avg”. Default is “avg”
                                - dropout (float): Dropout factor in [0, 1)
                                - activation (str): An activation function to apply “sigmoid”/”softmax” 
                                (could be None to return logits)
    """
    return SMP.DeepLabV3Plus(encoder_name='resnet50', encoder_depth=encoder_depth, encoder_weights=encoder_weights, encoder_output_stride=encoder_output_stride, 
                                decoder_channels=decoder_channels, decoder_atrous_rates=decoder_atrous_rates, in_channels=in_channels, classes=classes, 
                                activation=activation, upsampling=upsampling, aux_params=aux_params)

def deeplabv3plus_resnet101(encoder_name, encoder_depth, encoder_weights, encoder_output_stride, 
                            decoder_channels, decoder_atrous_rates, in_channels, classes, 
                            activation, upsampling, aux_params, **kwargs):

    return SMP.DeepLabV3Plus(encoder_name='resnet101', encoder_depth=encoder_depth, encoder_weights=encoder_weights, encoder_output_stride=encoder_output_stride, 
                                decoder_channels=decoder_channels, decoder_atrous_rates=decoder_atrous_rates, in_channels=in_channels, classes=classes, 
                                activation=activation, upsampling=upsampling, aux_params=aux_params)


if __name__ == "__main__":
    import torch
    from modelsummary import summary

    model = deeplabv3plus_resnet50(encoder_name='resnet50', encoder_depth=5, encoder_weights='imagenet', encoder_output_stride=16,
                                    decoder_channels=256, decoder_atrous_rates=(12 , 24, 36), in_channels=3, classes=2,
                                    activation=None, upsampling=4, aux_params=None)

    print('model output shape:', model(torch.rand(5, 3, 256, 256)).shape)
    print(summary(model, (3, 256, 256), device='cpu'))