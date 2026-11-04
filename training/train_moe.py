# Copyright (c) 2026 SB Intuitions Corp.
# SPDX-License-Identifier: MIT
"""Supervised fine-tuning entry point for the mixture-of-experts base model (Qwen3-30B-A3B).

FSDP-based counterpart of train.py (which targets dense models with DeepSpeed).
Differences: FSDP sharding via accelerate (DeepSpeed ZeRO-3 deadlocks on MoE
allgather when a rank routes zero tokens to an expert), the router auxiliary
loss is disabled to keep router logits from defeating gradient checkpointing,
gradient checkpointing uses the non-reentrant path, and batching keeps uniform
shapes across ranks (group_by_length=False). Otherwise identical: LoRA with
TRL's SFTTrainer, completion-only loss masking, early stopping, and
post-training adapter merge + checkpoint pruning (see train.py's docstring).
"""
import os
import sys
import shutil
import gc
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

# ==========================================
# 1. ENV VARS
# ==========================================
# Let Accelerate handle most distributed timeouts.
# We keep async error handling so failures crash fast instead of hanging.
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
import re
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    set_seed,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 3. HELPER: POST-TRAINING CLEANUP
# ==========================================
def cleanup_checkpoints(output_dir, best_checkpoint, logger):
    """
    After training keep exactly two directories:
      - checkpoint-best/  (best by eval loss, renamed from its numbered dir)
      - checkpoint-<N>/   (latest numbered checkpoint, for future training extension)
    Delete all other intermediate numbered checkpoints.
    """
    if not os.path.isdir(output_dir):
        return

    ckpt_re = re.compile(r"^checkpoint-\d+$")
    numbered = sorted(
        [d for d in os.listdir(output_dir)
         if ckpt_re.match(d) and os.path.isdir(os.path.join(output_dir, d))],
        key=lambda d: int(d.split("-")[1])
    )
    if not numbered:
        return

    latest_path = os.path.join(output_dir, numbered[-1])

    # Rename best numbered checkpoint → checkpoint-best
    best_stable = os.path.join(output_dir, "checkpoint-best")
    best_was_latest = False
    if best_checkpoint and os.path.isdir(best_checkpoint) and os.path.abspath(best_checkpoint) != os.path.abspath(best_stable):
        best_was_latest = (os.path.abspath(best_checkpoint) == os.path.abspath(latest_path))
        if os.path.isdir(best_stable):
            shutil.rmtree(best_stable)
        os.rename(best_checkpoint, best_stable)
        state_file = os.path.join(best_stable, "trainer_state.json")
        if os.path.isfile(state_file):
            with open(state_file) as f:
                state_data = json.load(f)
            state_data["best_model_checkpoint"] = best_stable
            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)
        logger.info(f"Best checkpoint renamed to: {best_stable}")

    # If best == latest, there is no separate latest to preserve
    keep_latest = not best_was_latest

    count = 0
    for d in numbered:
        full_path = os.path.join(output_dir, d)
        if keep_latest and os.path.abspath(full_path) == os.path.abspath(latest_path):
            logger.info(f"Preserving latest checkpoint: {full_path}")
            continue
        if os.path.isdir(full_path):
            try:
                shutil.rmtree(full_path)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {full_path}: {e}")
    logger.info(f"Cleanup complete: {count} intermediate checkpoint(s) removed.")

# ==========================================
# 4. CALLBACKS: MEMORY & MLFLOW LOGGING
# ==========================================
class MemoryClearCallback(TrainerCallback):
    """Free CUDA cache and Python garbage after every checkpoint save."""
    def on_save(self, args, state, control, **kwargs):
        """Release CUDA cache and Python garbage after each checkpoint save."""
        torch.cuda.empty_cache()
        gc.collect()

class PreSaveCacheClearCallback(TrainerCallback):
    """
    Clears the PyTorch memory cache at the end of every epoch.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        """Clear caches at the end of every epoch to limit memory fragmentation."""
        gc.collect()
        torch.cuda.empty_cache()

class MLflowRunCallback(TrainerCallback):
    """Single source of truth for MLflow lifecycle (start, log, end + resume).

    Replaces HF's auto-added MLflowCallback so there is exactly one run per
    training session regardless of what post-train events fire on_log.
    """
    def __init__(self):
        """Track whether this callback currently owns an active MLflow run."""
        self._active = False

    def on_train_begin(self, args, state, control, **kwargs):
        """Start (or resume) the one MLflow run for this training session, on rank 0 only."""
        if args.process_index != 0:
            return
        import mlflow
        run_id_file = os.path.join(args.output_dir, "mlflow_run_id.txt")
        resume_id = None
        if os.path.exists(run_id_file):
            with open(run_id_file) as f:
                resume_id = f.read().strip() or None
            if resume_id:
                print(f"\n>>> [MLflow] Resuming run {resume_id}")
        elif os.path.isdir(args.output_dir) and get_last_checkpoint(args.output_dir) is not None:
            print(f"\n>>> [MLflow WARNING] Checkpoint found but mlflow_run_id.txt missing!")
        run_name = os.path.basename(args.output_dir.rstrip("/"))
        mlflow.start_run(run_id=resume_id, run_name=run_name if not resume_id else None)
        self._active = True
        if not resume_id:
            new_id = mlflow.active_run().info.run_id
            with open(run_id_file, "w") as f:
                f.write(new_id)
            print(f"\n>>> [MLflow] New run '{run_name}' ({new_id}) saved to {run_id_file}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Forward numeric metrics to the active MLflow run (rank 0 only)."""
        if args.process_index != 0 or not self._active or not logs:
            return
        import mlflow
        if mlflow.active_run():
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """End the MLflow run when training finishes (rank 0 only)."""
        if args.process_index != 0:
            return
        self._active = False
        import mlflow
        if mlflow.active_run():
            mlflow.end_run()

# ==========================================
# 5. DATA COLLATOR
# ==========================================
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Mask every label up to and including the response template with -100.

    Finds the first occurrence of ``response_token_ids`` in each sequence and
    zeroes the loss for all preceding tokens (+ the template itself), so the
    model is trained on the assistant completion only. Sequences in which the
    template never occurs are fully masked.
    """
    def __init__(self, response_template: Union[str, List[int]], tokenizer: Any, mlm: bool = False, ignore_index: int = -100):
        """Encode the response template to token ids for the exact-match masking search."""
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.ignore_index = ignore_index
        if isinstance(self.response_template, list):
            self.response_token_ids = self.response_template
        else:
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        """Collate the batch, then mask prompt tokens out of the label tensor."""
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        template_tensor = torch.tensor(self.response_token_ids)
        n_template = len(self.response_token_ids)
        for i in range(len(labels)):
            response_token_ids_start_idx = None
            n_tokens = len(labels[i])
            for idx in range(n_tokens - n_template + 1):
                if torch.equal(labels[i][idx : idx + n_template], template_tensor):
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                labels[i, :] = self.ignore_index
            else:
                response_start = response_token_ids_start_idx + n_template
                labels[i, :response_start] = self.ignore_index
        batch["labels"] = labels
        return batch

# ==========================================
# 6. CONFIGURATION
# ==========================================
@dataclass
class ModelArguments:
    """Base-model selection and loading options (model id, dtype, flash attention)."""
    model_name_or_path: str = field(metadata={"help": "Hugging Face model ID or local path"})
    torch_dtype: str = field(default="bfloat16", metadata={"help": "bfloat16 or float16"})
    use_flash_attn: bool = field(default=True, metadata={"help": "Use Flash Attention 2"})

@dataclass
class LoraArguments:
    """LoRA adapter configuration (rank, alpha, dropout, target modules, bias)."""
    use_lora: bool = field(default=True, metadata={"help": "Use LoRA for parameter-efficient fine-tuning"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA scaling alpha; effective scale = alpha/r"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated linear layer names to adapt"}
    )
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias mode: none | all | lora_only"})

@dataclass
class DataArguments:
    """Training data paths and sequence settings (JSONL files, response template, context length)."""
    train_file: str = field(metadata={"help": "Path to train.jsonl"})
    val_file: str = field(default=None, metadata={"help": "Path to val.jsonl"})
    response_template: str = field(default="assistant", metadata={"help": "String marking assistant response start."})
    context_length: int = field(default=4096, metadata={"help": "Explicit Max sequence length"})

def setup_logging(training_args):
    """Configure stdout logging (+ train.log on rank 0) and return the module logger."""
    os.makedirs(training_args.output_dir, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout)]
    if training_args.process_index == 0:
        handlers.append(logging.FileHandler(os.path.join(training_args.output_dir, "train.log")))
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
    return logging.getLogger(__name__)

def verify_and_fix_tokenizer(tokenizer, response_template):
    """Ensure special markers in the response template tokenize atomically.

    If any <|...|> or [...] marker in the template encodes as several tokens,
    register all such markers as additional special tokens so the completion
    collator's exact-match search cannot silently miss them. Returns the
    (possibly patched) tokenizer; callers must resize embeddings when the
    vocabulary grew.
    """
    if not isinstance(response_template, str): return tokenizer
    patterns = re.findall(r"(<\|.*?\|>|\[.*?\])", response_template)
    tokens_to_check = list(set(patterns))
    if not tokens_to_check: return tokenizer
    needs_patch = False
    for t in tokens_to_check:
        encoded = tokenizer.encode(t, add_special_tokens=False)
        if len(encoded) > 1: needs_patch = True
    if needs_patch:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_check})
    return tokenizer

# ==========================================
# 7. MAIN FUNCTION
# ==========================================
def main():
    """Parse arguments, build the model/tokenizer/trainer, train, merge, and clean up.

    Same pipeline as train.py's main but FSDP-oriented: the router auxiliary
    loss is disabled, batch shapes stay uniform across ranks, and the adapter
    merge goes through Accelerate's unwrap with fsdp_use_orig_params=True.
    """
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    if data_args.response_template:
        data_args.response_template = data_args.response_template.replace("\\n", "\n")

    training_args.max_seq_length = data_args.context_length
    training_args.max_length = data_args.context_length

    training_args.dataloader_drop_last = True
    training_args.group_by_length = False  # FSDP requires uniform batch shapes across ranks

    training_args.ddp_timeout = 10800

    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # FSDP is configured via accelerate launch flags (--use_fsdp etc.)
    # to ensure AcceleratorState is initialized with FSDP from the start.
    # DeepSpeed ZeRO-3 is incompatible with MoE because allgather deadlocks
    # when some ranks skip experts that received zero routed tokens.

    training_args.save_safetensors = True
    training_args.load_best_model_at_end = True
    training_args.save_total_limit = None  # Keep all checkpoints; post-training cleanup retains best + latest

    if training_args.metric_for_best_model is None:
        training_args.metric_for_best_model = "loss"
        training_args.greater_is_better = False

    logger = setup_logging(training_args)
    set_seed(training_args.seed)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Checkpoint detected! Resuming training from: {last_checkpoint}")

    model_kwargs = {
        "torch_dtype": getattr(torch, model_args.torch_dtype),
        "trust_remote_code": True,
        "use_cache": False,
    }
    if model_args.use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    # MoE: disable router aux loss to save memory.
    # output_router_logits=True forces all 48 layers to hold router logits
    # (shape: [seq_len × num_experts] per layer) until the end of the forward
    # pass, defeating gradient checkpointing and causing OOM.
    # The router is already pre-trained; fine-tuning doesn't need
    # load-balancing regularization.
    model.config.output_router_logits = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    original_vocab_size = len(tokenizer)
    tokenizer = verify_and_fix_tokenizer(tokenizer, data_args.response_template)
    if len(tokenizer) != original_vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        target_dtype = getattr(torch, model_args.torch_dtype)
        for param in model.get_input_embeddings().parameters():
            if param.dtype != target_dtype: param.data = param.data.to(target_dtype)
        if model.get_output_embeddings() is not None:
            for param in model.get_output_embeddings().parameters():
                if param.dtype != target_dtype: param.data = param.data.to(target_dtype)

    if lora_args.use_lora:
        # Note: fsdp_config.yaml must have fsdp_use_orig_params: true (already set)
        modules_to_save = ["embed_tokens", "lm_head"] if len(tokenizer) != original_vocab_size else None
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules.split(","),
            bias=lora_args.lora_bias,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        logger.info(f"LoRA enabled: r={lora_args.lora_r}, alpha={lora_args.lora_alpha}, "
                    f"targets={lora_args.lora_target_modules}")
        model.print_trainable_parameters()

    data_files = {"train": data_args.train_file}
    if data_args.val_file: data_files["validation"] = data_args.val_file
    logger.info(f"Loading datasets from {data_args.train_file}")
    datasets = load_dataset("json", data_files=data_files)

    def formatting_func(example):
        """Render chat messages to text with the model's tokenizer chat template."""
        msgs = example['messages']
        if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], list):
            return [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in msgs]
        return [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)]

    collator = DataCollatorForCompletionOnlyLM(
        response_template=data_args.response_template,
        tokenizer=tokenizer
    )

    callbacks = [
        MemoryClearCallback, PreSaveCacheClearCallback,
        EarlyStoppingCallback(early_stopping_patience=5),
    ]
    # MLflow is opt-in: pass --report_to mlflow to track the run.
    use_mlflow = "mlflow" in (training_args.report_to or [])
    if use_mlflow:
        callbacks.append(MLflowRunCallback())

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        formatting_func=formatting_func,
        data_collator=collator,
        args=training_args,
        callbacks=callbacks,
    )

    # HF's Trainer auto-adds its own MLflowCallback via report_to; remove it so
    # MLflowRunCallback above is the single owner of the run lifecycle.
    if use_mlflow:
        from transformers.integrations import MLflowCallback as _HfMLflowCallback
        trainer.pop_callback(_HfMLflowCallback)

    logger.info("Starting Training...")

    try:
        # 1. Run the training loop
        resume_flag = True if last_checkpoint is not None else None
        train_result = trainer.train(resume_from_checkpoint=resume_flag)

        gc.collect()
        torch.cuda.empty_cache()

        # 2. Merge LoRA adapter into base model (required for vLLM evaluation)
        if lora_args.use_lora:
            logger.info("Merging LoRA adapter into base model (FSDP-aware)...")
            # Accelerate's unwrap_model handles FSDP; merge_and_unload works because
            # fsdp_use_orig_params=True keeps original parameter references intact.
            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            merged = unwrapped.merge_and_unload()
            trainer.model = merged
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("LoRA merge complete. Saving via FSDP-aware trainer.save_model()...")
            trainer.save_model(training_args.output_dir)
        else:
            logger.info("Saving Best Model (Sharded SafeTensors)...")
            trainer.save_model(training_args.output_dir)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # 3. Ensure all nodes finish writing their shards before moving on
        trainer.accelerator.wait_for_everyone()

        # 4. Clean up the intermediate checkpoint folders (Only Rank 0 runs this)
        if trainer.args.process_index == 0:
            cleanup_checkpoints(training_args.output_dir, trainer.state.best_model_checkpoint, logger)
            logger.info("\n" + "="*50)
            logger.info("Training, native sharding, and cleanup complete!")
            logger.info("="*50)

    except Exception as e:
        logger.error(f"Training interrupted: {e}")
        raise e

if __name__ == "__main__":
    main()
