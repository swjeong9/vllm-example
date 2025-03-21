from typing import List, Optional, Dict, Any, Tuple, Union
import torch
from array import array

from vllm.inputs.data import SingletonInputs
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (
    SequenceData, Sequence, SequenceGroupMetadata
)

TensorSlice = Tuple[torch.Tensor, int, int]
InputActivationType = Union[torch.Tensor, TensorSlice]

class PipelineSequenceData(SequenceData):
    """Pipeline sequence data with input activation type."""
    # SequenceData는 msgspec.Struct를 상속하므로 클래스 필드 방식으로 정의
    input_act: InputActivationType

    def get_input_activation(self) -> torch.Tensor:
        """Get the input activation for the pipeline."""
        return self.input_act
    
    def set_input_hidden(self, val: InputActivationType, is_prompt: bool):
        """Set the input hidden state for the pipeline."""
        self.input_act = val
        if is_prompt:
            assert len(self._output_token_ids) == 0

class PipelineSequence(Sequence):
    """Pipeline sequence with input activation type."""
    def __init__(
        self,
        seq_id: int,
        inputs: SingletonInputs,
        block_size: int,
        input_act: InputActivationType,  # 필수 인자로 변경
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        super().__init__(
            seq_id=seq_id,
            inputs=inputs,
            block_size=block_size,
            eos_token_id=eos_token_id,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request
        )
        # Recreate self.data attribute as PipelineSequenceData
        self.data = PipelineSequenceData(
            _prompt_token_ids=self.data._prompt_token_ids,
            _output_token_ids=self.data._output_token_ids,
            input_act=input_act
        )

    def set_input_hidden(self, val: InputActivationType, is_prompt: bool):
        """Set the input hidden state for the pipeline."""
        self.data.set_input_hidden(val, is_prompt)

class PipelineSequenceGroupMetadata(SequenceGroupMetadata):
    """Pipeline sequence group metadata."""
    
    def __post_init__(self):
        # 부모 클래스의 __post_init__ 호출
        super().__post_init__()
        self.seq_data: Dict[int, PipelineSequenceData]

    # 타입 명시를 위한 프로퍼티 추가
    @property
    def seq_data(self) -> Dict[int, PipelineSequenceData]:
        return super().seq_data  # type: ignore
    
    @seq_data.setter
    def seq_data(self, value: Dict[int, PipelineSequenceData]):
        super.__setattr__(self, "seq_data", value)  # msgspec.Struct 방식으로 설정

import dataclasses

@dataclasses.dataclass
class PipelineStageOut:
    output_tensor: torch.Tensor
    prompt_lens: List[int]
    is_prompt: bool
