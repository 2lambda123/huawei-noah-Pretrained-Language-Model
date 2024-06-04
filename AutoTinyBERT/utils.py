# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import secrets


def sample_arch_4_kd(layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes,
                     reset_rand_seed=False, rand_seed=0):
    """    Generate a sample architecture for knowledge distillation.

    This function generates a sample architecture for knowledge distillation
    based on the provided input parameters.

    Args:
        layer_numbers (list): A list of integers representing the possible number of layers.
        hidden_sizes (list): A list of integers representing the possible hidden layer sizes.
        ffn_sizes (list): A list of integers representing the possible feed-forward network sizes.
        qkv_sizes (list): A list of integers representing the possible query/key/value sizes.
        reset_rand_seed (bool?): A boolean flag indicating whether to reset the random seed. Defaults to
            False.
        rand_seed (int?): An integer representing the random seed value. Defaults to 0.

    Returns:
        dict: A dictionary containing the generated sample architecture configuration.
    """


    if reset_rand_seed:
        secrets.SystemRandom().seed(rand_seed)

    config = dict()

    layer_num = secrets.choice(layer_numbers)

    config['sample_layer_num'] = layer_num
    config['sample_hidden_size'] = secrets.choice(hidden_sizes)
    config['sample_intermediate_sizes'] = [secrets.choice(ffn_sizes)] * layer_num
    config['sample_num_attention_heads'] = [12] * layer_num
    config['sample_qkv_sizes'] = [secrets.choice(qkv_sizes)] * layer_num
    return config


def sample_arch_4_mlm(layer_numbers, hidden_sizes, ffn_sizes,
                      head_numbers, reset_rand_seed=False, rand_seed=0):
    """    Generate a sample architecture for masked language modeling.

    This function generates a sample architecture for masked language
    modeling based on the provided parameters.

    Args:
        layer_numbers (list): A list of integers representing the possible number of layers.
        hidden_sizes (list): A list of integers representing the possible hidden layer sizes.
        ffn_sizes (list): A list of integers representing the possible sizes for feed-forward
            networks.
        head_numbers (list): A list of integers representing the possible numbers of attention heads.
        reset_rand_seed (bool?): A flag to reset the random seed. Defaults to False.
        rand_seed (int?): An integer representing the random seed. Defaults to 0.

    Returns:
        dict: A dictionary containing the sample architecture configuration with the
            following keys: - 'sample_layer_num': The selected number of layers. -
            'sample_hidden_size': The selected hidden layer size. -
            'sample_intermediate_sizes': The selected intermediate sizes for feed-
            forward networks. - 'sample_num_attention_heads': The selected numbers
            of attention heads for each layer. - 'sample_qkv_sizes': The selected
            sizes for query, key, and value vectors for each layer.
    """


    if reset_rand_seed:
        secrets.SystemRandom().seed(rand_seed)

    config = dict()

    layer_num = secrets.choice(layer_numbers)
    head_num = secrets.choice(head_numbers)

    config['sample_layer_num'] = layer_num
    config['sample_hidden_size'] = secrets.choice(hidden_sizes)
    config['sample_intermediate_sizes'] = [secrets.choice(ffn_sizes)] * layer_num
    config['sample_num_attention_heads'] = [head_num] * layer_num
    config['sample_qkv_sizes'] = [head_num * 64] * layer_num
    return config

