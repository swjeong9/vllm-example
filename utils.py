import ray
from typing import Dict, List
import os
from vllm.logger import init_logger

logger = init_logger(__name__)

def create_placement_group_and_bundle_indices(node_rank_mapping: Dict[str, List[int]]):
    logger.info(f"Creating placement group and bundle indices for node rank mapping: {node_rank_mapping}")

    if not ray.is_initialized():
        ray.init(address="auto")

    # 전체 rank 수 계산 및 rank-IP 매핑 생성
    total_ranks = 0
    rank_to_ip = {}
    for ip, ranks in node_rank_mapping.items():
        num_ranks_on_ip = len(ranks)
        total_ranks += num_ranks_on_ip
        for rank in ranks:
            if rank in rank_to_ip:
                 raise ValueError(f"Rank {rank} is assigned to multiple IPs. Ranks must be unique.")
            rank_to_ip[rank] = ip

    # Rank가 0부터 total_ranks-1까지 연속적인지 확인
    if set(rank_to_ip.keys()) != set(range(total_ranks)):
        raise ValueError(f"Ranks must be contiguous from 0 to {total_ranks - 1}. Found ranks: {sorted(rank_to_ip.keys())}")

    # 각 rank에 대해 placement group spec 생성
    placement_group_specs: List[Dict[str, float]] = []
    for rank in range(total_ranks):
        ip = rank_to_ip[rank]
        placement_group_specs.append({
            'GPU': 1,
            f"node:{ip}": 0.001 # 특정 노드를 사용하겠다는 의미
        })
    
    # strategy 를 STRICT_SPREAD 로 설정하면 모든 랭크가 다 다른 노드에 분배되어야 함.
    # 따라서 ray.get 에서 무한대기를 하는 상황이 발생한다. PACK 으로 변경하자.
    placement_group = ray.util.placement_group(placement_group_specs, strategy="PACK")
    ray.get(placement_group.ready())

    # 생성된 번들(bundle)과 할당된 노드 IP 매핑
    bundle_to_node = {} # { bundle_idx: node_ip }
    for bundle_id, bundle in enumerate(placement_group.bundle_specs):
        for resource_key in bundle:
            if resource_key.startswith("node:"):
                node_ip = resource_key[5:] # 'node:172.31.16.230' -> '172.31.16.230'
                bundle_to_node[bundle_id] = node_ip
                break # 노드 IP 찾으면 다음 번들로 넘어감

    # Rank 순서에 맞게 번들 인덱스 할당
    bundle_indices = [None] * total_ranks
    # 각 IP별로 아직 할당되지 않은 rank 리스트 (정렬된 상태)
    remaining_ranks_for_ip = {ip: sorted(ranks) for ip, ranks in node_rank_mapping.items()}
    assigned_bundles = set() # 이미 할당된 번들 추적

    for bundle_idx in range(len(placement_group.bundle_specs)):
        if bundle_idx not in bundle_to_node:
            raise RuntimeError(f"Bundle {bundle_idx} is not assigned to any node.")

        # bundle index 와 해당 bundle index 의 node ip 를 추출한다.
        node_ip = bundle_to_node[bundle_idx]

        if node_ip in remaining_ranks_for_ip and remaining_ranks_for_ip[node_ip]:
            # 해당 IP에 할당해야 할 가장 낮은 rank를 가져옴
            assigned_rank = remaining_ranks_for_ip[node_ip].pop(0)
            if bundle_indices[assigned_rank] is not None:
                 raise RuntimeError(f"Rank {assigned_rank} is already assigned to bundle {bundle_indices[assigned_rank]}. Trying to assign bundle {bundle_idx}.")
            bundle_indices[assigned_rank] = str(bundle_idx)
            assigned_bundles.add(bundle_idx) # 할당된 번들로 기록

            # 해당 IP의 모든 rank가 할당되었으면 딕셔너리에서 제거
            if not remaining_ranks_for_ip[node_ip]:
                del remaining_ranks_for_ip[node_ip]
        else:
            # 해당 번들에 매칭되는 rank가 없는 경우 (로직 오류 또는 PG 생성 문제)
             raise RuntimeError(f"Could not find a rank assignment for bundle {bundle_idx} on node {node_ip}. Remaining ranks: {remaining_ranks_for_ip}")


    # 모든 rank가 번들에 할당되었는지 확인
    if None in bundle_indices:
        unassigned_ranks = [i for i, b in enumerate(bundle_indices) if b is None]
        raise RuntimeError(f"Could not assign bundles to all ranks. Unassigned ranks: {unassigned_ranks}. Assignments: {bundle_indices}")
    if remaining_ranks_for_ip:
         raise RuntimeError(f"Some ranks were not assigned bundles: {remaining_ranks_for_ip}")
    if len(assigned_bundles) != len(placement_group.bundle_specs):
        raise RuntimeError(f"Number of assigned bundles ({len(assigned_bundles)}) does not match total bundles ({len(placement_group.bundle_specs)}).")


    # 환경 변수 설정 및 출력
    os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(bundle_indices)
    print(f"VLLM_RAY_BUNDLE_INDICES is setted to {os.environ['VLLM_RAY_BUNDLE_INDICES']}")
    print(f"bundle specs : {placement_group.bundle_specs}")

    return placement_group




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