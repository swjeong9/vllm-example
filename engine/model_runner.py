# The original code is in https://github.com/Thesys-lab/Helix-ASPLOS25/blob/master/llm_sys/engine/model_runner.py
from typing import List, Optional, Sequence, Tuple, Union, Any, Dict, Type
import time
import torch
import torch.nn as nn
import dataclasses
import contextlib

from vllm.config import VllmConfig
from vllm.model_executor.model_loader import get_model
from vllm.worker.model_runner import GPUModelRunnerBase, ModelInputForGPUWithSamplingMetadata
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import DeviceMemoryProfiler
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger

from .sequence import (PipelineStageOut, PipelineSequenceGroupMetadata, PipelineSequenceData)

logger = init_logger(__name__)

class LayerwiseModelRunner(GPUModelRunnerBase):
    """ModelRunner that supports executing specific layers of a model independently.
    
    This is useful for pipeline parallelism where different layers can be distributed
    across different devices.
    """
    
    def __init__(
            self, 
            vllm_config: VllmConfig,
            layer_ids: List[int],
            **kwargs: Any,
    ) -> None:
        super().__init__(vllm_config, **kwargs)
        self.layer_ids = layer_ids
        
        # Total number of layers in the model
        total_layer_num = getattr(self.model_config.hf_text_config, "num_hidden_layers", 0)
        self.last_layer_id = total_layer_num - 1
        self.num_total_layer = total_layer_num
        
        # Dictionary to hold layer-specific models
        self.models: Dict[int, nn.Module] = {layer_id: None for layer_id in self.layer_ids}

    def load_model(self) -> None:
        """Load model weights layer by layer to optimize memory usage."""
        with DeviceMemoryProfiler(self.device) as m:
            for layer_id in self.layer_ids:
                # Configure model to load specific layer
                self.model_config.hf_config.layer_id = layer_id
                self.models[layer_id] = get_model(self.vllm_config)
                
        self.model_memory_usage = m.consumed_memory
        logger.info(f"Loading model weights took "
                    f"{self.model_memory_usage / float(2**30):.4f} GB")
    
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[PipelineSequenceGroupMetadata],
        kv_cache: torch.Tensor,
        layer_id: int,
        **kwargs,
    ) -> Optional[Union[SamplerOutput, PipelineStageOut, IntermediateTensors]]:
        """Execute a specific layer of the model.
        
        Args:
            seq_group_metadata_list: List of sequence group metadata
            kv_cache: KV cache tensor
            layer_id: ID of the layer to execute
            
        Returns:
            For non-final layers: PipelineStageOut containing hidden states
            For final layer: SamplerOutput containing generated tokens
            
        Note: This function adapts the old layerwise execution model to the new vLLM interface.
        """
        # Check if the layer is loaded
        if layer_id not in self.models:
            raise ValueError(f"Layer {layer_id} is not loaded in this ModelRunner")
        
        # Use the standard model input preparation method
        model_input = self.prepare_model_input(seq_group_metadata_list, 
                                              finished_requests_ids=None)
        
        # Get the model for this layer
        model_executable = self.models[layer_id]
        
        # Determine if this is the first layer - needed for adapting inputs
        is_first_layer = (layer_id == 0)
        
        # Apply LoRA if configured
        if self.lora_config and model_input.lora_requests and model_input.lora_mapping:
            self.set_active_loras(model_input.lora_requests, model_input.lora_mapping)
            
        # Apply prompt adapters if configured
        if (self.prompt_adapter_config and model_input.prompt_adapter_requests 
            and model_input.prompt_adapter_mapping):
            self.set_active_prompt_adapters(
                model_input.prompt_adapter_requests,
                model_input.prompt_adapter_mapping)
        
        # Initialize attention state for this forward pass
        self.attn_state.begin_forward(model_input)
        
        # Prepare multi-modal kwargs if present
        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        
        # Stateful model handling
        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}
        
        # Prepare execution arguments
        execute_model_kwargs = {
            "positions": model_input.input_positions,
            "kv_cache": kv_cache,
            "attn_metadata": model_input.attn_metadata,
            **seqlen_agnostic_kwargs,
            **multi_modal_kwargs,
        }
        
        # Add inputs based on layer position
        if is_first_layer:
            # First layer takes token IDs as input
            execute_model_kwargs["input_ids"] = model_input.input_tokens
        else:
            # Other layers take hidden states
            # For layerwise execution, we need to handle the hidden states from previous layer
            # In the actual implementation, this would come from the previous layer's output
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None:
                raise ValueError("Hidden states must be provided for non-first layers")
            execute_model_kwargs["hidden_states"] = hidden_states
            
        # Execute the model with proper context
        with set_forward_context(model_input.attn_metadata, 
                                self.vllm_config, 
                                model_input.virtual_engine):
            output_hidden_states = model_executable(**execute_model_kwargs)
            
        # Determine next steps based on whether this is the final layer
        is_final_layer = (layer_id == self.last_layer_id or 
                          layer_id == max(self.layer_ids))
        is_prompt = seq_group_metadata_list[0].is_prompt if seq_group_metadata_list else False
        
        if is_final_layer and model_input.sampling_metadata is not None:
            # For the final layer, compute logits and sample tokens
            logits = model_executable.compute_logits(output_hidden_states, 
                                                    model_input.sampling_metadata)
            
            # Only perform sampling on the driver worker
            if not self.is_driver_worker:
                return None
                
            # Sample the next token
            output = model_executable.sample(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
            
            # Add hidden states if requested
            if self.return_hidden_states:
                if is_prompt:
                    output.prefill_hidden_states = output_hidden_states
                output.hidden_states = output_hidden_states
                
            return output
        else:
            # For non-final layers, return intermediate hidden states
            # that will be passed to the next layer
            return PipelineStageOut(
                hidden_states=output_hidden_states,
                is_prompt=is_prompt
            )
            
    @torch.inference_mode()
    def profile_run(self) -> None:
        """Run profiling to measure memory usage."""
        # Use the standard profile mechanism, but with a specific layer
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        
        # Choose a representative layer to profile
        if self.last_layer_id in self.layer_ids:
            profile_layer_id = self.last_layer_id  # Last layer has the most complexity
        else:
            profile_layer_id = self.layer_ids[0]  # Default to first available
        
        # Adjust dummy run to profile only the selected layer
        self._dummy_run(max_num_batched_tokens, max_num_seqs, layer_id=profile_layer_id)
        
    def _dummy_run(self,
                  max_num_batched_tokens: int,
                  max_num_seqs: int = 1,
                  layer_id: Optional[int] = None) -> None:
        """Run a dummy model execution for profiling.
        
        Override the parent method to include layer-specific execution.
        """
        with self.set_in_profile_run():
            # Call parent implementation to prepare dummy data and inputs
            super()._dummy_run(max_num_batched_tokens, max_num_seqs)

