# The original code is in https://github.com/Thesys-lab/Helix-ASPLOS25/blob/master/llm_sys/engine/llama.py

from typing import List, Optional
from transformers import LlamaConfig

import torch
import torch.nn as nn

from vllm.model_executor.layers.linear import (LinearMethodBase)
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata


class LlamaPipelineStage(nn.Module):
    # 이것을 사용하는 이유는 Q, K, V 를 생성하는 과정이 모두 Linear 이기 때문이다.
    # Q, K, V 를 순차적으로 생성하는 것 보다는 matrix 를 concat 한 뒤 한 번에 계산하는 것이
    # GPU 를 사용하는 입장에서 더 효율적이기 때문이다.
    # 하지만 현재는 notimplemented 로 둔다
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # 임베딩은 여러 종류가 있다. 모델에서 사용되는 임베딩 관련 모듈들의 이름을 표준화 하여 관리한다.
    embedding_modules = {
        "embed_tokens": "input_embeddings", # 입력 임베딩 (토큰 -> 벡터) 을 담당하는 이름
        "lm_head": "output_embeddings",     # 출력 임베딩 (벡터 -> 토큰) 을 담당하는 이름
    }
    embedding_padding_modules = ["lm_head"] # 출력 임베딩 모듈 (vocabulary projection layer) 에서 padding 이 적용됨
                                            # 왜 필요할까? : GPU 연산 효율성 때문에 2의 배수로 padding 을 적용함

    
    def __init__(
            self,
            config: LlamaConfig,
            linear_method: Optional[LinearMethodBase] = None,
            lora_config=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.vocab_size = config.vocab_size

        # 원래의 LlamaConfig 에는 layer_id 같은 것이 없다. 입력으로 따로 넣어주어야 한다.
        # e.g.) config.layer_id = 0 등
        layer_id = config.layer_id
        self.layer_id = layer_id

        self.decoder_layer = LlamaDecoderLayer(config, linear_method) # 1개의 디코더 레이어

        # 첫 번째 레이어일 경우 토큰을 vector 로 변환시키는 작업을 거쳐야 한다.
        if layer_id == 0:
            # VocabParallelEmbedding 은 임베딩을 담당하는 클래스이다.
            # Parallel 이 붙은 이유는 Tensor 병렬화를 적용할 경우 vocabulary 를 각 shard 에 나누어 
            # 저장 후 병렬적으로 처리를 맡게 하는 특징을 가지고 있다.
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        
        # 마지막 레이어일 경우 vector 를 토큰으로 변환시키는 작업을 거쳐야 한다.
        # 또한 elif 로 처리하지 않은 이유는 하나의 stage 가
        # 첫 번째 레이어와 마지막 레이어를 모두 포함할 수 있기 때문이다.
        if layer_id == self.config.num_hidden_layers - 1:
            self.unpadded_vocab_size = config.vocab_size
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            # lm_head 는 language model head 를 의미한다.
            # output 의 vector size 는 (batch_size, sequence_length, hidden_size) 이다.
            # 이를 다시 logit 으로 바꿔주는 것이 LM Head 인데 즉, hidden_size -> vocab_size 로 변환하는 것이다.
            # logits = hidden_states @ W + b   (W 의 크기는 hidden_size x vocab_size)
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                if not lora_config else lora_config.vocab_padding_size,
            )

            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
    
    def forward(self, **kwargs):
        positions = kwargs["positions"]
        kv_cache = kwargs["kv_cache"]
        attn_metadata = kwargs["attn_metadata"]

        # 첫 번째 layer 의 경우에는 input token 을 임베딩 한뒤 해당 입력을 넣어준다.
        if self.layer_id == 0:
            input_ids = kwargs["input_ids"]
            inputs_embeds = (kwargs["input_embeds"] if "input_embeds" in kwargs else None)
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                # token -> vector (hidden_states)
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
            hidden_states, residual = self.decoder_layer(
                positions,
                hidden_states,
                kv_cache,
                attn_metadata,
                residual,
            )
        else:
            hidden_states = kwargs["hidden_states"]
            residual = None
            hidden_states, residual = self.decoder_layer(
                positions,
                hidden_states,
                kv_cache,
                attn_metadata,
                residual,
            )
            # 마지막 레이어의 경우 normalization 을 해주자
            if self.layer_id == self.config.num_hidden_layers - 1:
                hidden_states = self.norm(hidden_states)
        return hidden_states

    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    # 이부분에서 packed_modules_mapping 을 사용하게 될 것이다.
    # vllm 의 Llama를 참고할 것
    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        raise NotImplementedError()

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states, sampling_metadata)
        return logits
    
    def sample(self,
               logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


from vllm.model_executor.models import ModelRegistry, llama

registry = ModelRegistry()
# def register_model(
#     self,
#     model_arch: str,
#     model_cls: Union[Type[nn.Module], str],
# ) -> None:
registry.register_model("LlamaForCausalLM", ("llama", "LlamaPipelineStage"))
llama.LlamaPipelineStage = LlamaPipelineStage