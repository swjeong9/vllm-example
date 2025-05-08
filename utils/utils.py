
def get_flops_hexgen(input_sequence_length: int,
                     output_sequence_length: int,
                     hidden_dim_size: int,
                     tp_degree: int = 1):
    return 24 * (input_sequence_length + output_sequence_length) * (hidden_dim_size**2 / tp_degree)

def get_memory_scan_cost_hexgen(output_sequence_length: int,
                                data_byte_size: int,
                                hidden_dim_size: int,
                                tp_degree: int = 1):
    # 어처피 hidden_dim_size 가 tp_degree 로 나누어 떨어지지 않으면 안됨
    assert hidden_dim_size % tp_degree == 0
    return 12 * hidden_dim_size**2 / tp_degree * data_byte_size * output_sequence_length

def get_computation_latency_hexgen(input_sequence_length: int,
                                   output_sequence_length: int,
                                   hidden_dim_size: int,
                                   data_byte_size: int,
                                   device_FLOPS: int,
                                   device_memory_bandwidth: int,
                                   tp_degree: int = 1):
    flops_per_layer = get_flops_hexgen(input_sequence_length, output_sequence_length, hidden_dim_size, tp_degree)
    memory_scan_cost_per_layer = get_memory_scan_cost_hexgen(output_sequence_length, data_byte_size, hidden_dim_size, tp_degree)
    return flops_per_layer / device_FLOPS, memory_scan_cost_per_layer / device_memory_bandwidth


def get_flops(input_sequence_length: int,
              output_sequence_length: int,
              hidden_dim_size: int,
              num_attention_heads: int,
              num_key_value_heads: int,
              intermediate_dim_size: int,
              tp_degree: int = 1):
    assert hidden_dim_size % num_attention_heads == 0
    key_value_hidden_size = hidden_dim_size // num_attention_heads * num_key_value_heads
    prefill_flops = input_sequence_length * hidden_dim_size * \
            (4*hidden_dim_size + 4*input_sequence_length + 4*key_value_hidden_size + 6*intermediate_dim_size)
    decode_flops = output_sequence_length * hidden_dim_size * \
            (4*hidden_dim_size + 4*input_sequence_length + 4*key_value_hidden_size + 6*intermediate_dim_size + 2*output_sequence_length + 2)
    return (prefill_flops + decode_flops) / tp_degree

def get_memory_scan_cost(input_sequence_length: int,
                         output_sequence_length: int,
                         hidden_dim_size: int,
                         num_attention_heads: int,
                         num_key_value_heads: int,
                         intermediate_dim_size: int,
                         data_byte_size: int,
                         tp_degree: int = 1):
    assert hidden_dim_size % num_attention_heads == 0
    key_value_hidden_size = hidden_dim_size // num_attention_heads * num_key_value_heads
    prefill_memory_scan_cost = 2 * input_sequence_length * hidden_dim_size
    prefill_memory_scan_cost += (2*hidden_dim_size**2 + 2*hidden_dim_size*key_value_hidden_size + input_sequence_length*hidden_dim_size \
            + 2*input_sequence_length*key_value_hidden_size + 3*hidden_dim_size*intermediate_dim_size \
            + 3*input_sequence_length*intermediate_dim_size) / tp_degree
    decode_memory_scan_cost = 2*hidden_dim_size + ((2*hidden_dim_size**2 + 2*hidden_dim_size*key_value_hidden_size + hidden_dim_size \
                                                   + 2*input_sequence_length*key_value_hidden_size + 3*hidden_dim_size*intermediate_dim_size \
                                                   + 3*intermediate_dim_size + output_sequence_length*key_value_hidden_size \
                                                   + key_value_hidden_size) / tp_degree)
    decode_memory_scan_cost *= output_sequence_length

    return (prefill_memory_scan_cost + decode_memory_scan_cost) * data_byte_size

def get_computation_latency(input_sequence_length: int,
                            output_sequence_length: int,
                            hidden_dim_size: int,
                            num_attention_heads: int,
                            num_key_value_heads: int,
                            intermediate_dim_size: int,
                            data_byte_size: int,
                            device_FLOPS: int,
                            device_memory_bandwidth: int,
                            tp_degree: int = 1):
    flops = get_flops(input_sequence_length, output_sequence_length, hidden_dim_size, num_attention_heads, num_key_value_heads, intermediate_dim_size, tp_degree)
    memory_scan_cost = get_memory_scan_cost(input_sequence_length, output_sequence_length, hidden_dim_size, num_attention_heads, num_key_value_heads, intermediate_dim_size, data_byte_size, tp_degree)
    return flops / device_FLOPS, memory_scan_cost / device_memory_bandwidth