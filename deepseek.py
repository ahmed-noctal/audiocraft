from typing import Dict, Optional, Tuple, Any, Union
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import logging

# Set up logging
logger = logging.getLogger(__name__)
def load_model_with_quantization_fallback(
        model_name: str = "deepseek-ai/DeepSeek-R1",
        trust_remote_code: bool = True,
        device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
        **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Model loaded successfully with original configuration")
        return model, tokenizer
    except ValueError as e:
        if "Unknown quantization type" in str(e):
            logger.warning(
                "Quantization type not supported directly. "
                "Attempting to load without quantization..."
            )

            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            if hasattr(config, "quantization_config"):
                delattr(config, "quantization_config")

            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    config=config,
                    trust_remote_code=trust_remote_code,
                    device_map=device_map,
                    **kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code
                )
                logger.info("Model loaded successfully without quantization")
                return model, tokenizer

            except Exception as inner_e:
                logger.error(f"Failed to load model without quantization: {str(inner_e)}")
                raise
        else:
            logger.error(f"Unexpected error during model loading: {str(e)}")
            raise


model, tokenizer = load_model_with_quantization_fallback(
    model_name="deepseek-ai/DeepSeek-R1",
    trust_remote_code=True,
    device_map="auto"
)

# Test the model
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))