"""Configuration objects and model discovery.

Configs are plain dataclasses with YAML round-tripping. Nothing here is model-specific: a
new checkpoint is added by pointing :func:`discover_model` at its directory, which reads the
HuggingFace metadata already sitting next to the weights and writes a config file.

That matters because the models arrive as ``.safetensors`` directories on the SSD with no
guarantee about layout -- single-file or sharded, quantized or not, text-only or
multimodal -- and the harness has to make sense of all of those without hand-editing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------------------
# YAML helpers -- kept tolerant so configs load with or without PyYAML installed
# --------------------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".json",):
        return json.loads(text)
    try:
        import yaml

        return yaml.safe_load(text) or {}
    except ImportError as exc:                                   # pragma: no cover
        raise RuntimeError(
            f"Reading {path} needs PyYAML. Install it, or use a .json config."
        ) from exc


def _dump_yaml(data: dict, path: Path) -> Path:
    """Write a config, in the format the path's suffix asks for.

    The suffix has to be honoured rather than always emitting YAML: :func:`_load_yaml` reads
    a ``.json`` path with ``json.loads``, so writing YAML there produced a file that this
    module could write but not read back.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        return path
    try:
        import yaml

        path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
                        encoding="utf-8")
    except ImportError:                                          # pragma: no cover
        path = path.with_suffix(".json")
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _from_dict(cls: type, data: dict) -> Any:
    """Construct a dataclass from a dict, recursing into nested dataclass fields.

    Unknown keys are ignored rather than raising, so a config written by a newer version of
    the harness still loads in an older one.
    """
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------

#: Quantization markers found in HF config files, mapped to a canonical short name. The
#: scheme drives bytes-per-weight in the analytic model and is a primary comparison axis.
QUANT_MARKERS = {
    "mixed_precision": "mixed",
    "awq": "awq",
    "gptq": "gptq",
    "nvfp4": "nvfp4",
    "fp8": "fp8",
    "fp4": "fp4",
    "modelopt": "modelopt",
    "bitsandbytes": "bitsandbytes",
    "compressed-tensors": "compressed-tensors",
}

#: Approximate bits per stored weight, used to predict decode-phase DRAM traffic before
#: measuring it. The measured value is the ground truth; this is the expectation we check
#: it against, and a large gap means the scoping or the derivation is wrong.
BITS_PER_WEIGHT = {
    None: 16, "bfloat16": 16, "float16": 16, "float32": 32,
    "awq": 4, "gptq": 4, "nvfp4": 4, "fp4": 4, "fp8": 8,
    "bitsandbytes": 8, "modelopt": 8, "compressed-tensors": 8,
    # A mixed-precision checkpoint has no single bits-per-weight. 8 is a placeholder for the
    # parameter-count estimate only; every byte figure that matters uses the on-disk size,
    # which is exact regardless of how the precision is distributed across layers.
    "mixed": 8,
}

#: KV-cache element size in bytes, by the checkpoint's declared KV quantization. This is
#: separate from weight quantization and often differs from it -- an FP8 KV cache halves the
#: cache's bytes, which changes decode-phase traffic directly and would otherwise make the
#: expected-versus-measured check disagree for a reason that has nothing to do with the
#: measurement.
KV_DTYPE_BYTES = {
    None: 2, "fp8": 1, "float8": 1, "fp4": 1, "nvfp4": 1, "int8": 1,
    "fp16": 2, "bf16": 2, "bfloat16": 2, "float16": 2, "auto": 2,
}


@dataclass
class ModelConfig:
    """One model under test."""

    name: str
    path: str

    #: Compute dtype requested at load. Ignored for pre-quantized checkpoints, which carry
    #: their own storage dtype.
    dtype: str = "bfloat16"

    #: Attention kernel: "sdpa", "eager", or "flash_attention_2". A major axis for shared
    #: memory and L2 behaviour -- eager materialises the full attention matrix, sdpa does not.
    attn_implementation: str = "sdpa"

    trust_remote_code: bool = False
    task: str = "text-generation"
    device_map: str | None = None
    tokenizer_path: str | None = None
    max_position_embeddings: int | None = None

    # ---- discovered metadata, filled by discover_model ----
    architecture: str = ""
    model_type: str = ""
    quantization: str | None = None
    #: Declared KV-cache quantization, if any. Independent of weight quantization.
    kv_quantization: str | None = None
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    #: Whether the input embedding and the LM head are the same tensor. Recorded because it
    #: decouples the checkpoint's size from the model's resident size: a tied checkpoint may
    #: still store both copies on disk, and the loaded model then holds one. That gap is a
    #: whole embedding matrix, which on a large-vocabulary model is a quarter of the total.
    tie_word_embeddings: bool = False

    # ---- mixture-of-experts ----------------------------------------------------------
    #: Routed experts per MoE layer, and how many of them each token actually activates.
    #:
    #: These are the difference between a model's size and its cost. A dense model reads
    #: every weight to emit a token; a top-8-of-128 MoE reads the router, the attention
    #: block, any shared expert, and 8 experts -- so its decode step moves a small fraction
    #: of the checkpoint. Predicting decode traffic from the checkpoint size on such a model
    #: overstates it by roughly num_experts/top_k, which is an order of magnitude.
    num_experts: int = 0
    num_experts_per_token: int = 0
    #: Experts every token passes through regardless of routing, where the architecture has
    #: them. They count as active, not routed.
    num_shared_experts: int = 0
    #: Layers that are MoE rather than dense. Several architectures interleave them, or keep
    #: the first N layers dense, so this is not always equal to num_layers.
    num_moe_layers: int = 0

    param_count: int = 0
    weight_bytes_on_disk: int = 0
    safetensors_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def bits_per_weight(self) -> int:
        return BITS_PER_WEIGHT.get(self.quantization or self.dtype, 16)

    @property
    def kv_dtype_bytes(self) -> int:
        """Bytes per KV-cache element, from the checkpoint's declared KV quantization."""
        return KV_DTYPE_BYTES.get((self.kv_quantization or "").lower() or None, 2)

    @property
    def benchmarkable(self) -> bool:
        """Whether this checkpoint has weights to load at all."""
        return bool(self.safetensors_files and self.weight_bytes_on_disk)

    def embedding_matrix_bytes(self) -> int:
        """Size of one vocabulary x hidden embedding matrix at this model's weight precision."""
        if not (self.vocab_size and self.hidden_size):
            return 0
        return int(self.vocab_size * self.hidden_size * self.bits_per_weight / 8)

    def estimated_weight_bytes(self) -> int:
        """Predicted bytes of weights a decode step must read.

        Prefers the on-disk size, which already reflects the real storage format including
        quantization scales and zero-points, over a parameter-count estimate.

        This is a **fallback**. The measured resident size is better wherever it is
        available, because on-disk size answers a slightly different question: a checkpoint
        with tied embeddings can still store ``embed_tokens.weight`` and ``lm_head.weight``
        as two separate tensors, and the loaded model then holds one. A decode step reads
        that matrix once -- for the output projection; the input side is a single-row
        gather -- so counting it twice overstates what the step must move.
        """
        if self.weight_bytes_on_disk:
            return self.weight_bytes_on_disk
        return int(self.param_count * self.bits_per_weight / 8)

    #: Set by discovery when the checkpoint physically stores an ``lm_head`` tensor. Together
    #: with ``tie_word_embeddings`` this is what decides whether the on-disk size counts the
    #: embedding matrix twice.
    stores_lm_head: bool | None = None

    @property
    def is_moe(self) -> bool:
        return bool(self.num_experts and self.num_experts_per_token)

    @property
    def expert_activation_ratio(self) -> float | None:
        """Fraction of each MoE layer's expert weights a single token routes through.

        ``None`` for a dense model, where the question does not apply.
        """
        if not self.is_moe:
            return None
        active = self.num_experts_per_token + self.num_shared_experts
        return min(1.0, active / self.num_experts)

    def attention_bytes(self) -> int:
        """Bytes of attention projection weights across all layers.

        Q, K, V and O for every block. Grouped-query attention is accounted for -- K and V
        project to ``num_key_value_heads`` rather than the full head count, which on these
        models is a 4-8x difference and would badly overstate the total if ignored.
        """
        head_dim = self.head_dim or (
            self.hidden_size // self.num_attention_heads if self.num_attention_heads else 0
        )
        if not (self.num_layers and self.hidden_size and head_dim
                and self.num_attention_heads):
            return 0
        kv_heads = self.num_key_value_heads or self.num_attention_heads
        # q: hidden x (q_heads*head_dim), o: (q_heads*head_dim) x hidden  -> 2 * q
        # k and v: hidden x (kv_heads*head_dim) each                      -> 2 * kv
        per_layer = self.hidden_size * head_dim * (
            2 * self.num_attention_heads + 2 * kv_heads
        )
        return int(per_layer * self.num_layers * self.bits_per_weight / 8)

    def active_weight_bytes(self) -> int:
        """Weight bytes a single decode step actually reads.

        For a dense model this is every weight, and the answer is just
        :meth:`estimated_weight_bytes`.

        For a mixture of experts it is not. Only ``top_k`` of ``num_experts`` are routed to
        per token, so the expert parameters -- which on these architectures are the large
        majority of the checkpoint -- are read at a fraction of their stored size. Comparing
        measured decode traffic against the *whole* checkpoint would report a 30B A3B model
        as moving a tenth of what theory predicts, and the harness would blame its own NVTX
        scoping for what is actually the model working as designed.

        The split matters more than it might seem. Attention, norms and the LM head are read
        in full on every step no matter how the router behaves, and on a sparse model they
        are most of what a decode step actually moves. Folding them into the expert-scaled
        remainder -- the obvious shortcut -- understates the expectation by around a quarter
        on these checkpoints, enough to push a healthy run toward the edge of the verdict
        band. So the always-active part is sized directly and only the expert remainder is
        scaled.
        """
        total = self.estimated_weight_bytes()
        ratio = self.expert_activation_ratio
        if ratio is None or not total:
            return total

        if self.tie_word_embeddings and self.stores_lm_head:
            # The checkpoint stores two copies but a step reads the matrix once.
            total -= self.embedding_matrix_bytes()

        # Read in full every step, regardless of routing.
        always_active = self.embedding_matrix_bytes() + self.attention_bytes()
        expert_part = max(0, total - always_active)
        return int(always_active + expert_part * ratio)

    def on_disk_may_double_count_embeddings(self) -> bool:
        """Whether the on-disk size overstates the weights a decode step must read.

        True only when the checkpoint declares tied embeddings *and* is actually storing a
        separate ``lm_head`` tensor. Both halves are needed and neither can be inferred from
        file size: the size is what we are trying to explain, so deriving the test from it is
        circular. The tensor list in the safetensors header answers it directly.
        """
        return bool(self.tie_word_embeddings and self.stores_lm_head)

    def kv_cache_bytes(
        self, seq_len: int, batch_size: int = 1, kv_dtype_bytes: int | None = None
    ) -> int:
        """Analytic KV-cache size for a given context length.

        2 (K and V) x layers x kv_heads x head_dim x seq_len x batch x dtype_bytes.
        Uses ``num_key_value_heads`` so grouped-query attention is accounted for correctly --
        assuming full attention heads would overstate this several-fold on modern models.
        """
        if kv_dtype_bytes is None:
            kv_dtype_bytes = self.kv_dtype_bytes
        if not (self.num_layers and self.num_key_value_heads):
            return 0
        head_dim = self.head_dim or (
            self.hidden_size // self.num_attention_heads if self.num_attention_heads else 0
        )
        if not head_dim:
            return 0
        return (
            2 * self.num_layers * self.num_key_value_heads * head_dim
            * seq_len * batch_size * kv_dtype_bytes
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        return _dump_yaml(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        return _from_dict(cls, _load_yaml(Path(path)))


def _sum_safetensors(directory: Path) -> tuple[list[str], int]:
    """List safetensors shards and their combined size.

    Sharded checkpoints are described by ``model.safetensors.index.json``; when that exists
    we trust its file list, because a directory can also hold leftover single-file copies of
    the same weights (as one of the checkpoints on this machine does), and summing every
    ``.safetensors`` blindly would double-count them.
    """
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
            names = sorted(set(index.get("weight_map", {}).values()))
            paths = [directory / n for n in names]
            if paths and all(p.exists() for p in paths):
                return [p.name for p in paths], sum(p.stat().st_size for p in paths)
        except (OSError, json.JSONDecodeError):
            pass

    files = sorted(directory.glob("*.safetensors"))
    # Prefer a canonical shard set over stray duplicates left beside it.
    sharded = [f for f in files if re.match(r"model-\d+-of-\d+\.safetensors$", f.name)]
    if sharded:
        files = sharded
    elif (directory / "model.safetensors").exists():
        files = [directory / "model.safetensors"]
    return [f.name for f in files], sum(f.stat().st_size for f in files)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _first_int(config: dict, *keys: str) -> int:
    """First positive integer among several candidate keys.

    Every vendor spells the mixture-of-experts fields differently -- ``num_experts``,
    ``num_local_experts``, ``n_routed_experts`` all mean the same thing across Qwen, Gemma
    and Nemotron checkpoints -- and reading the wrong one silently yields a dense model.
    """
    for key in keys:
        value = config.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return 0


def _detect_moe(text_config: dict) -> dict:
    """Read the mixture-of-experts shape, tolerating each vendor's spelling."""
    experts = _first_int(
        text_config, "num_experts", "num_local_experts", "n_routed_experts",
        "moe_num_experts", "num_routed_experts",
    )
    top_k = _first_int(
        text_config, "num_experts_per_tok", "moe_topk", "num_experts_per_token",
        "top_k_experts", "moe_top_k", "experts_per_token",
    )
    shared = _first_int(
        text_config, "n_shared_experts", "num_shared_experts", "moe_num_shared_experts",
        "shared_expert_intermediate_size_count",
    )
    # Architectures that interleave dense and MoE blocks say so; otherwise every layer with
    # experts is one.
    layers = _first_int(text_config, "num_hidden_layers")
    first_dense = _first_int(text_config, "first_k_dense_replace", "num_dense_layers")
    step = _first_int(text_config, "moe_layer_freq") or 1
    moe_layers = 0
    if experts and layers:
        moe_layers = max(0, (layers - first_dense + step - 1) // step)

    return {
        "num_experts": experts,
        "num_experts_per_token": top_k,
        "num_shared_experts": shared,
        "num_moe_layers": moe_layers if experts else 0,
    }


def _safetensors_keys(directory: Path, files: list[str]) -> set[str]:
    """Tensor names in a checkpoint, without loading any weights.

    A safetensors file begins with a little-endian u64 header length followed by that many
    bytes of JSON naming every tensor, so the key list costs one small read per shard. The
    sharded index, where present, already lists them and is read instead.

    Reading the names matters because file size alone cannot distinguish a checkpoint that
    ties its embeddings and stores one copy from one that ties them and stores two -- and
    those differ by a whole embedding matrix in what a decode step is predicted to read.
    """
    index = _read_json(directory / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if isinstance(weight_map, dict) and weight_map:
        return set(weight_map)

    keys: set[str] = set()
    for name in files:
        try:
            with open(directory / name, "rb") as handle:
                header_len = int.from_bytes(handle.read(8), "little")
                # A sane header is kilobytes; a wild value means this is not safetensors.
                if not 0 < header_len < 100 * 1024 * 1024:
                    continue
                header = json.loads(handle.read(header_len).decode("utf-8"))
            keys.update(k for k in header if k != "__metadata__")
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    return keys


def _detect_quantization(
    config: dict, directory: Path
) -> tuple[str | None, str | None, list[str]]:
    """Identify the weight and KV-cache quantization schemes from HF metadata.

    Returns ``(weight_scheme, kv_scheme, notes)``. The two are reported separately because
    they genuinely differ in practice -- one checkpoint here declares MIXED_PRECISION weights
    alongside an FP8 KV cache -- and only the KV scheme changes the analytic cache size.

    A repository name is not evidence: one checkpoint on this machine is named ``...-NVFP4``
    while its own ``hf_quant_config.json`` declares ``MIXED_PRECISION``. Only the metadata is
    trusted.
    """
    notes: list[str] = []
    kv_scheme: str | None = None
    blob = json.dumps(config).lower()

    # KV-cache quantization is declared separately from weight quantization.
    for source in (config.get("quantization_config") or {}, _read_json(directory / "hf_quant_config.json")):
        if not isinstance(source, dict):
            continue
        for container in (source, source.get("quantization") or {}):
            if not isinstance(container, dict):
                continue
            raw_kv = container.get("kv_cache_quant_algo") or container.get("kv_cache_dtype")
            if raw_kv and str(raw_kv).lower() not in ("null", "none"):
                kv_scheme = str(raw_kv).lower()
                notes.append(f"KV cache is quantized ({kv_scheme}); analytic cache size uses it")
                break
        if kv_scheme:
            break

    quant_config = config.get("quantization_config") or {}
    if isinstance(quant_config, dict):
        # quant_algo before quant_method, deliberately. They are different things and a
        # checkpoint can carry both: quant_algo names the *scheme* (what precision the
        # weights are actually stored at), while quant_method names the *toolkit* that
        # produced the checkpoint. One model here declares quant_method="modelopt" alongside
        # quant_algo="MIXED_PRECISION" -- preferring the method would label a mixed-precision
        # checkpoint as uniformly 8-bit, which is exactly the kind of silent error a
        # quantization comparison must not make.
        algo = str(quant_config.get("quant_algo") or "").lower()
        toolkit = str(quant_config.get("quant_method") or "").lower()
        method = algo or toolkit
        if algo and toolkit and algo != toolkit:
            notes.append(
                f"quant_algo={algo} produced by quant_method={toolkit}; the algorithm "
                "determines precision, so the algorithm is what is reported"
            )
        for marker, canonical in QUANT_MARKERS.items():
            if marker in method:
                bits = quant_config.get("bits") or quant_config.get("w_bit")
                if bits:
                    notes.append(f"quantization_config reports {bits}-bit weights")
                return canonical, kv_scheme, notes

    if (directory / "hf_quant_config.json").exists():
        try:
            hf_quant = json.loads((directory / "hf_quant_config.json").read_text())
            algo = str(
                hf_quant.get("quantization", {}).get("quant_algo")
                or hf_quant.get("quant_algo")
                or ""
            ).lower()
            for marker, canonical in QUANT_MARKERS.items():
                if marker in algo:
                    notes.append(f"detected via hf_quant_config.json (quant_algo={algo})")
                    if canonical == "mixed":
                        notes.append(
                            "precision varies per layer, so bits-per-weight is indeterminate; "
                            "byte figures use the exact on-disk size instead"
                        )
                    return canonical, kv_scheme, notes
            if algo:
                notes.append(f"hf_quant_config.json reports unrecognised quant_algo={algo}")
        except (OSError, json.JSONDecodeError):
            notes.append("hf_quant_config.json present but unreadable")

    for marker, canonical in QUANT_MARKERS.items():
        if f'"{marker}"' in blob or f"_{marker}_" in blob:
            notes.append(f"inferred from a '{marker}' mention in config.json")
            return canonical, kv_scheme, notes

    return None, kv_scheme, notes


def discover_model(
    path: str | Path, name: str | None = None, **overrides
) -> ModelConfig:
    """Build a :class:`ModelConfig` by reading a checkpoint directory.

    Handles single-file and sharded layouts, quantized checkpoints, and multimodal configs
    whose text-model parameters are nested under ``text_config``/``llm_config``.
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Not a model directory: {directory}")

    config_path = directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json in {directory}. Point this at a HuggingFace-format checkpoint "
            "directory, not at a bare .safetensors file."
        )

    config = json.loads(config_path.read_text())
    notes: list[str] = []

    # Multimodal checkpoints nest the language model's shape one level down. Reading the
    # top level would report a vision tower's dimensions as if they were the LLM's.
    text_config = config
    for key in ("text_config", "llm_config", "language_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and nested.get("num_hidden_layers"):
            text_config = nested
            notes.append(f"transformer shape read from nested '{key}' (multimodal checkpoint)")
            break

    files, total_bytes = _sum_safetensors(directory)
    if not files:
        notes.append(
            "NO WEIGHTS FOUND -- this directory has config and tokenizer files but no "
            ".safetensors. It cannot be benchmarked as-is; the download is probably "
            "incomplete, or the weights are in another format."
        )

    quantization, kv_quantization, quant_notes = _detect_quantization(config, directory)
    notes.extend(quant_notes)

    architectures = config.get("architectures") or []
    num_heads = text_config.get("num_attention_heads", 0) or 0
    hidden = text_config.get("hidden_size", 0) or 0

    model = ModelConfig(
        name=name or directory.name,
        path=str(directory),
        architecture=architectures[0] if architectures else "",
        model_type=config.get("model_type", ""),
        quantization=quantization,
        kv_quantization=kv_quantization,
        num_layers=text_config.get("num_hidden_layers", 0) or 0,
        hidden_size=hidden,
        num_attention_heads=num_heads,
        num_key_value_heads=text_config.get("num_key_value_heads", num_heads) or num_heads,
        head_dim=text_config.get("head_dim", 0) or (hidden // num_heads if num_heads else 0),
        vocab_size=text_config.get("vocab_size", 0) or 0,
        intermediate_size=text_config.get("intermediate_size", 0) or 0,
        tie_word_embeddings=bool(
            text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False))
        ),
        **_detect_moe(text_config),
        weight_bytes_on_disk=total_bytes,
        safetensors_files=files,
        max_position_embeddings=text_config.get("max_position_embeddings"),
        notes=notes,
    )

    if config.get("auto_map") or any(p.suffix == ".py" for p in directory.glob("*.py")):
        model.trust_remote_code = True
        model.notes.append(
            "custom modelling code present -- trust_remote_code enabled automatically"
        )

    if text_config.get("torch_dtype"):
        model.dtype = str(text_config["torch_dtype"])

    if model.weight_bytes_on_disk and model.bits_per_weight:
        model.param_count = int(model.weight_bytes_on_disk * 8 / model.bits_per_weight)

    if files:
        keys = _safetensors_keys(directory, files)
        if keys:
            model.stores_lm_head = any(k.endswith("lm_head.weight") for k in keys)

    if model.is_moe:
        ratio = model.expert_activation_ratio or 0
        model.notes.append(
            f"Mixture of experts: {model.num_experts} experts per layer, "
            f"{model.num_experts_per_token} routed per token"
            + (f" plus {model.num_shared_experts} shared" if model.num_shared_experts else "")
            + f" ({ratio:.1%} of expert weights active). A decode step reads roughly "
            f"{model.active_weight_bytes() / 1e9:,.1f} GB of the "
            f"{model.estimated_weight_bytes() / 1e9:,.1f} GB checkpoint, so decode traffic "
            "is compared against the active figure -- against the full checkpoint it would "
            "read as an order of magnitude too low."
        )

    if model.on_disk_may_double_count_embeddings():
        model.notes.append(
            "This checkpoint declares tied word embeddings and stores lm_head.weight anyway, "
            "so the on-disk size counts the embedding matrix twice. The loaded model holds "
            "one copy, so its resident size reads about "
            f"{model.embedding_matrix_bytes() / 1e6:,.0f} MB below the on-disk figure. That "
            "is expected, not a discrepancy -- and the resident figure is the one a decode "
            "step's traffic must be compared against, since the step reads that matrix once."
        )

    for key, value in overrides.items():
        if hasattr(model, key) and value is not None:
            setattr(model, key, value)

    return model


# --------------------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------------------


@dataclass
class WorkloadConfig:
    """What to run against the model.

    Prefill and decode are measured separately throughout, because they sit at opposite ends
    of the memory story: prefill is compute-bound and reuses weights across many tokens,
    while decode reads the entire weight set to produce a single token and is bound almost
    entirely by memory bandwidth.
    """

    name: str = "default"
    kind: str = "text-generation"

    prompt_tokens: int = 512
    generate_tokens: int = 64
    batch_size: int = 1

    #: Iterations discarded before measurement -- covers autotuning, lazy module init and
    #: allocator warmup. Without these the first measured step is wildly unrepresentative.
    warmup_iters: int = 3

    #: Measured repeats. Reported as median with IQR rather than a mean, so one slow
    #: iteration from a thermal excursion does not skew the headline number.
    repeat: int = 5

    #: "synthetic" generates a deterministic token sequence; otherwise a path to a text file.
    prompt_source: str = "synthetic"

    #: Greedy by default. Sampling adds run-to-run variance that would show up as noise in
    #: the memory numbers for no analytical benefit.
    do_sample: bool = False
    temperature: float = 1.0
    seed: int = 1234

    #: Annotate individual transformer blocks with NVTX. Adds hook overhead, so it is off
    #: for timing runs and on for attribution runs.
    annotate_layers: bool = False

    #: Multimodal placeholders, honoured by the multimodal workload when it is implemented.
    image_count: int = 0
    image_size: tuple[int, int] = (448, 448)
    audio_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        return _dump_yaml(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> WorkloadConfig:
        return _from_dict(cls, _load_yaml(Path(path)))


# --------------------------------------------------------------------------------------
# Profiler settings
# --------------------------------------------------------------------------------------


@dataclass
class NsysConfig:
    """Nsight Systems collection settings."""

    enabled: bool = True

    #: cublas/cudnn tracing attributes time to the library that issued it, which is how the
    #: report can say "GEMM" rather than listing mangled cutlass kernel names.
    trace: str = "cuda,nvtx,cublas,cudnn,osrt"

    cuda_memory_usage: bool = True
    cuda_um_cpu_page_faults: bool = True
    cuda_um_gpu_page_faults: bool = True

    #: Hardware counter sampling. The set is chosen from the chip name by platform detection;
    #: on GB10 it carries clocks, activity and warp occupancy but no bandwidth rows.
    gpu_metrics: bool = True
    gpu_metrics_frequency: int = 10_000

    #: "node" resolves kernels inside CUDA graphs individually instead of collapsing a whole
    #: graph launch into one opaque entry.
    cuda_graph_trace: str = "node"

    #: Restrict the trace to the cudaProfilerStart/Stop region, skipping load and warmup.
    capture_range: str = "cudaProfilerApi"
    capture_range_end: str = "stop"

    #: Python backtrace sampling. Needs perf_event access, so it stays off by default here.
    python_sampling: bool = False

    timeout_s: int = 3600


@dataclass
class NcuConfig:
    """Nsight Compute collection settings."""

    enabled: bool = True

    #: 1 = curated memory metric list over every kernel in one decode step.
    #: 2 = full sections on the heaviest kernels.
    #: 3 = adds source-level attribution.
    tiers: tuple[int, ...] = (1, 2)

    #: How many kernels tier 2 deep-dives, ranked by measured time from the nsys pass.
    top_n_kernels: int = 8

    #: NVTX range that scopes collection. One decode step yields one instance of each unique
    #: kernel -- full coverage at a fraction of the cost of profiling the whole generation.
    nvtx_scope: str = "nsbench.decode_step"

    #: "base" asks ncu to lock clocks. GB10 exposes no supported clock list through
    #: nvidia-smi, so the lock may be a no-op; the harness records the measured GPC clock
    #: from the nsys pass instead of trusting it.
    clock_control: str = "base"

    #: Flush caches between replay passes so each pass sees the same cold state. Turning this
    #: off makes L2 hit rates unrepresentative of the real, un-replayed execution.
    cache_control: str = "all"

    replay_mode: str = "kernel"

    #: Cap on kernel launches profiled per collection, passed to ncu as --launch-count.
    #:
    #: This is a runaway guard, not a sampling knob: ncu simply stops after this many
    #: launches, so a cap below the real kernel count silently truncates the totals rather
    #: than sampling them. The runner detects that and flags the collection as partial.
    #:
    #: Sizing it: an eager transformers decode step runs far more kernels than the layer
    #: count suggests -- roughly 55-60 launches per transformer block once normalisation,
    #: rotary embedding and elementwise ops are counted, so a 28-layer model is around 1600
    #: launches per step. Budget roughly one minute of collection per 100 kernels at tier 1.
    max_kernels: int = 4000

    #: Whether the runner may raise :attr:`max_kernels` to fit a deeper model.
    #:
    #: The cap has to serve two opposite jobs. Left alone it should scale with the model,
    #: because a value sized for a 28-layer dense model silently truncates a 52-layer
    #: mixture of experts and costs that run its physics check. But it must also stay
    #: honourable as an explicit instruction: the smoke test deliberately sets a cap far
    #: below the real kernel count to check the plumbing in minutes, and an "auto" mode that
    #: overrode that would make the fast path slow again.
    #:
    #: So: automatic while nobody has said otherwise, exact once someone has. Passing
    #: ``--max-kernels`` clears this.
    auto_launch_cap: bool = True

    timeout_s: int = 7200


@dataclass
class ProfileConfig:
    nsys: NsysConfig = field(default_factory=NsysConfig)
    ncu: NcuConfig = field(default_factory=NcuConfig)

    #: Unprofiled timing pass. The only source of honest latency and throughput, since both
    #: profilers perturb execution -- ncu especially, since it replays each kernel.
    baseline: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> ProfileConfig:
        data = _load_yaml(Path(path))
        cfg = cls()
        if "nsys" in data:
            cfg.nsys = _from_dict(NsysConfig, data["nsys"])
        if "ncu" in data:
            cfg.ncu = _from_dict(NcuConfig, data["ncu"])
        if "baseline" in data:
            cfg.baseline = bool(data["baseline"])
        return cfg

    def save(self, path: str | Path) -> Path:
        return _dump_yaml(self.to_dict(), Path(path))


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Everything needed to execute and identify one benchmark run."""

    model: ModelConfig
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    backend: str = "hf"
    output_root: str = "runs"
    tag: str = ""

    def run_id(self, timestamp: str) -> str:
        """Directory name for this run: sortable by time, readable at a glance."""
        parts = [timestamp, _slug(self.model.name), _slug(self.workload.name), self.backend]
        if self.tag:
            parts.append(_slug(self.tag))
        return "__".join(parts)

    def fingerprint(self) -> str:
        """Stable hash over the settings that affect results.

        ``output_root`` and ``tag`` are excluded so two runs that differ only in where they
        were written still compare as the same configuration.
        """
        payload = {
            "model": {
                k: v for k, v in self.model.to_dict().items()
                if k not in ("notes", "safetensors_files")
            },
            "workload": self.workload.to_dict(),
            "profile": self.profile.to_dict(),
            "backend": self.backend,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_dict(),
            "workload": self.workload.to_dict(),
            "profile": self.profile.to_dict(),
            "backend": self.backend,
            "output_root": self.output_root,
            "tag": self.tag,
            "fingerprint": self.fingerprint(),
        }

    def save(self, path: str | Path) -> Path:
        return _dump_yaml(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        data = _load_yaml(Path(path))
        run = cls(model=_from_dict(ModelConfig, data["model"]))
        if "workload" in data:
            run.workload = _from_dict(WorkloadConfig, data["workload"])
        if "profile" in data:
            profile = ProfileConfig()
            if "nsys" in data["profile"]:
                profile.nsys = _from_dict(NsysConfig, data["profile"]["nsys"])
            if "ncu" in data["profile"]:
                profile.ncu = _from_dict(NcuConfig, data["profile"]["ncu"])
            profile.baseline = data["profile"].get("baseline", True)
            run.profile = profile
        run.backend = data.get("backend", "hf")
        run.output_root = data.get("output_root", "runs")
        run.tag = data.get("tag", "")
        return run


def _slug(text: str) -> str:
    """Filesystem-safe token for use in run directory names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "unnamed"
