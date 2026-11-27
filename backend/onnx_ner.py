"""Run an exported ONNX (INT8) NER model behind the same interface as a Hugging Face token-classification pipeline.

export_onnx_ner.py produces the model folder. This module loads it with onnxruntime only (no `onnx` package needed) and
wraps it so that main.run_ner() and the evaluation scripts can call it exactly like the PyTorch pipeline:
    nlp = load_pipeline("models/bert-base-NER-onnx")
    nlp(["Rahul works at Infosys."], batch_size=8)   ->  [[{"entity_group": "PER", "word": "Rahul", "start": 0, ...}]]
The tokenizer, the label map and the word-level aggregation ("first") are the Hugging Face ones, so only the neural
network itself is replaced.
"""
import os
from pathlib import Path


def load_pipeline(model_dir, file="model_int8.onnx", threads=None):
    import onnxruntime as ort
    import torch
    from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, pipeline
    from transformers.modeling_outputs import TokenClassifierOutput

    model_dir = Path(model_dir)
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_dir / file), options, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    class _OnnxTokenClassifier(PreTrainedModel):
        config_class = type(config)

        def __init__(self, cfg):
            super().__init__(cfg)
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)      # gives the module a device and dtype

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **_):
            feeds = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
            logits = session.run(None, {n: feeds[n].cpu().numpy() for n in input_names})[0]
            return TokenClassifierOutput(logits=torch.from_numpy(logits))

    # the pipeline checks the class name against the architectures it supports for this task
    _OnnxTokenClassifier.__name__ = (config.architectures or ["BertForTokenClassification"])[0]
    return pipeline("ner", model=_OnnxTokenClassifier(config).eval(), tokenizer=tokenizer, aggregation_strategy="first")


def default_dir() -> str:
    return os.getenv("NER_ONNX_DIR", "").strip()
