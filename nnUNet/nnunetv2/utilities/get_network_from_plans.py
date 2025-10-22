import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        import dynamic_network_architectures
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None and 'deep_supervision' not in arch_kwargs.keys():
        arch_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    # def count_parameters(model):

    #     total_trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     total_params = sum(p.numel() for p in model.parameters())

    #     # imageencoder_trained_params = sum(p.numel() for p in model.sam_model.image_encoder.parameters() if p.requires_grad)
    #     # imageencoder_params = sum(p.numel() for p in model.sam_model.image_encoder.parameters())

    #     # promptencoder_trained_params = sum(p.numel() for p in model.sam_model.prompt_encoder.parameters() if p.requires_grad)
    #     # promptencoder_params = sum(p.numel() for p in model.sam_model.prompt_encoder.parameters())

    #     # imagedecoder_trained_params = sum(p.numel() for p in model.sam_model.mask_decoder.parameters() if p.requires_grad)
    #     # imagedecoder_params = sum(p.numel() for p in model.sam_model.mask_decoder.parameters())

    #     return [total_trained_params, total_params]

    # # (total_trained_params, total_params, 
    # # imageencoder_trained_params, imageencoder_params, 
    # # promptencoder_trained_params, promptencoder_params, 
    # # imagedecoder_params, imagedecoder_params) = count_parameters(network)

    # params_list = count_parameters(network)
    # params_list =  [i / (1024*1024) for i in params_list]

    # # params_list = [i / (1024*1024) for i in [total_trained_params, total_params, 
    # # imageencoder_trained_params, imageencoder_params, 
    # # promptencoder_trained_params, promptencoder_params, 
    # # imagedecoder_params, imagedecoder_params]]

    # [total_trained_params, total_params] = params_list

    # print("total_trained_params = {}, total_params = {}, \n".format(total_trained_params, total_params,))

    # assert False

    return network
