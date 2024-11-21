from typing import Any

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print('Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion')

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from PIL import Image as PILImage

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    num_outputs: int = 1,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
                num_outputs=num_outputs,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all(
            [isinstance(p, str) or isinstance(p, Image) or isinstance(p, PILImage.Image) for p in prompt]
        )
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
            # n=1  # Gemini only supports 1 output for now
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response



# added by Yu Gu
def openai_messages_to_claude(messages: list) -> list:
    # convert the messages from openai format to claude format
    new_messages = []
    system_message = None
    for h in messages:
            if h['role'] == 'agent':
                h['role'] = 'assistant'
            # convert the openai format to claude format
            if isinstance(h['content'], list):
                for i, c in enumerate(h['content']):
                    if isinstance(c, dict) and c.get('type') == 'image_url':
                        new_c = {}
                        new_c['type'] = 'image'
                        new_c['source'] = {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": c['image_url']['url'].split(",")[1],
                        }
                        h['content'][i] = new_c
            if h['role'] != 'system':
                new_messages.append(h)
            else:
                system_message = h['content']   # todo: this does not handle multiple system messages

    return new_messages, system_message