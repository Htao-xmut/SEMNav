import logging
import os
import base64
import torch
import numpy as np
import google.generativeai as genai
import cv2
import threading

from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline
from utils import append_mime_tag, encode_image_b64, resize_image_if_needed


class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent.
    This class should be extended to implement specific VLMs.
    """

    def __init_subclass__(cls, **kwargs):
        """p26 修复① (专家评审方案): import 时机签名契约强制。

        #26c 事故: plain_text 加到了 OllamaVLM (line 370), 实弹用的
        QwenVLClient (line 709) 没有 → 三局 18 次关键 VLM 分析全崩,
        单测假绿 (静态断言只 grep 'plain_text' 字符串)。@abstractmethod
        只查"方法被覆写", 不查签名兼容 (Python 无此机制) — 本钩子用
        inspect 逐参比对, 子类缺基类参数时 **类定义即抛 TypeError**,
        烧 API 之前就炸。宁可报错, 不可静默适配 (不用 **kwargs 糊)。
        """
        super().__init_subclass__(**kwargs)
        if 'call_chat' in cls.__dict__:
            import inspect as _inspect
            base_params = set(_inspect.signature(
                VLM.call_chat).parameters) - {'self'}
            sub_params = set(_inspect.signature(
                cls.__dict__['call_chat']).parameters) - {'self'}
            missing = base_params - sub_params
            if missing:
                raise TypeError(
                    f"{cls.__name__}.call_chat 契约违约: 缺基类参数 "
                    f"{sorted(missing)} (基类签名参数: {sorted(base_params)}) "
                    f"— 修子类签名, 不要加 **kwargs 掩盖")

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def call_chat(self, history: int, images: list[np.array], text_prompt: str,
                  plain_text: bool = False):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        plain_text : bool
            True = 本次调用要自由文本 + 规定行 (FIRST_DIRECTION: /
            SCAN_SUSPICIOUS: 等六图扫描协议) → 子类不得注入 JSON-only
            系统指令 (#26c 根因)。
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class GeminiVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-1.5-flash", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # Configure generation parameters
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        }

        self.spend = 0
        self.cost_per_input_token = 0.075 / 1_000_000 if 'flash' in self.name else 1.25 / 1_000_000
        self.cost_per_output_token = 0.3 / 1_000_000 if 'flash' in self.name else 5 / 1_000_000

        # Initialize Gemini model and chat session
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            system_instruction=system_instruction
        )
        self.session = self.model.start_chat(history=[])

    def call_chat(self, history: int, images: list[np.array], text_prompt: str,
                  plain_text: bool = False):
        """
        Perform context-aware inference with the Gemini model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.session.send_message([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

            # Manage history length based on the number of past steps to keep
            if history == 0:
                self.session = self.model.start_chat(history=[])
            elif len(self.session.history) > 2 * history:
                self.session.history = self.session.history[-2 * history:]

        except Exception as e:
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.session.history) > 1:
            self.model.rewind()

    def reset(self):
        """
        Reset the chat history.
        """
        self.session = self.model.start_chat(history=[])

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.model.generate_content([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

        except Exception as e:
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend


class OpenAIVLM(VLM):
    """
    An implementation using models served via OpenAI API.
    """

    def __init__(self, model="gpt-4o-latest", system_instruction=None, max_image_res=None):
        """
        Initialize the OpenAI model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        from openai import OpenAI
        self.name = model
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model
        self.history = [] 
        self.max_image_res = max_image_res


    def call_chat(self, history: int, images: list[np.array], text_prompt: str,
                  plain_text: bool = False):
        """
        Perform context-aware inference with the OpenAI model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history + messages,
            )
            self.history.append(messages[0]) # append user message
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": response.choices[0].message.content}]}) # append response

            # Manage history length based on the number of past steps to keep
            if len(self.history) > 2 * history:
                self.history = self.history[-2 * history:]

        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content


    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.history) > 1:
            self.history = self.history[:-2]

    def reset(self):
        """
        Reset the chat history.
        """
        self.history = []


    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        text_contents = [{
            "type": "text",
            "text": text_prompt
        }]
        image_contents = self._image_contents_from_images(images)
        messages = [{"role": "user", "content": text_contents + image_contents}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

        except Exception as e:
            logging.error(f"OPENAI API ERROR: {e}")
            return "OPENAI API ERROR"

        return response.choices[0].message.content


    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        NOTE: not implemented
        """
        return 0


    def _image_contents_from_images(self, images: list[np.ndarray]):
        image_contents = [
            {
                "type": "image_url",
                "image_url":
                {
                    "url": append_mime_tag(encode_image_b64(Image.fromarray(image[:, :, :3], mode='RGB'))) \
                    if self.max_image_res is None else append_mime_tag(encode_image_b64(
                        resize_image_if_needed(Image.fromarray(image[:, :, :3], mode='RGB'), self.max_image_res)))
                }
            }
            for image in images
        ]
        return image_contents


class OllamaVLM(VLM):
    """
    An implementation using local Ollama API for Qwen and other models.
    Supports both text-only and vision-language models via Ollama.
    """

    def __init__(self, model="qwen3.5:9b", system_instruction=None, max_image_res=None):
        """
        Initialize the Ollama model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used (e.g., "qwen3.5:9b", "llava:latest").
        system_instruction : str, optional
            System instructions for model behavior.
        max_image_res : int, optional
            Maximum image resolution. If None, no resizing is applied.
        """
        # Import requests at module level to avoid repeated imports
        import requests
        
        self.name = model
        self.requests = requests  # Store as instance variable
        self.api_url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
        self.chat_api_url = os.environ.get("OLLAMA_CHAT_API_URL", "http://localhost:11434/api/chat")
        self.model = model
        self.system_instruction = system_instruction
        self.history = []
        self.max_image_res = max_image_res
        self.request_timeout = 180  # seconds
        self._encode_lock = threading.Lock() if threading else None

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode image to base64 string for Ollama vision models."""
        from PIL import Image
        import io
        import base64
        
        img_pil = Image.fromarray(image[:, :, :3], mode='RGB')
        
        # Resize if needed
        if self.max_image_res is not None:
            img_pil = resize_image_if_needed(img_pil, self.max_image_res)
        
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=85)
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded

    def call_chat(self, history: int, images: list[np.array], text_prompt: str,
                  plain_text: bool = False):
        """
        Perform context-aware inference with the Ollama model.
        Uses /api/chat endpoint for better multi-turn conversation support.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        plain_text : bool
            True = 本次调用要自由文本 + 规定行 (FIRST_DIRECTION: /
            SCAN_SUSPICIOUS: 等六图扫描协议) → 不注入 JSON-only 系统
            指令。p16 三局实弹: SCAN_ 行零解析的根因即此压制 (所有
            调用默认带 "Respond ONLY with JSON")。
        """
        # Build messages for chat API
        messages = []

        # Check if this is a LLaVA model (special handling required)
        is_llava = 'llava' in self.model.lower()

        # Add system instruction if provided (SKIP for LLaVA - it doesn't support system prompts well)
        if self.system_instruction and not is_llava and not plain_text:
            messages.append({
                "role": "system",
                "content": self.system_instruction
            })
        
        # Add conversation history
        if history > 0 and len(self.history) > 0:
            messages.extend(self.history[-2*history:])
        
        # Build user message with images
        # Note: Different models have different requirements
        is_vision_model = any(keyword in self.model.lower() for keyword in ['llava', 'bakllava', 'moondream', 'cogvlm', 'qwen'])
        
        if images and is_vision_model:
            if is_llava:
                # LLaVA in /api/chat requires base64 images in the 'images' array within the message
                # Encode all images to base64
                image_base64_list = []
                for img in images:
                    try:
                        base64_img = self._encode_image_to_base64(img)
                        image_base64_list.append(base64_img)
                    except Exception as e:
                        logging.error(f"Error encoding image for LLaVA: {e}")
                
                # For LLaVA, put text in content and images in separate 'images' field
                user_message = {
                    "role": "user",
                    "content": text_prompt,
                    "images": image_base64_list  # Array of base64 strings
                }
                messages.append(user_message)
            else:
                # For other vision models (Qwen, etc.), use content array format
                user_content = []
                for img in images:
                    try:
                        base64_img = self._encode_image_to_base64(img)
                        user_content.append({
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_img}"
                        })
                    except Exception as e:
                        logging.error(f"Error encoding image: {e}")
                
                user_content.append({"type": "text", "text": text_prompt})
                user_message_content = user_content
                
                messages.append({
                    "role": "user",
                    "content": user_message_content
                })
        else:
            # For text-only models or when no images, use string format
            full_prompt = text_prompt
            if images:
                full_prompt = f"[Image observation: {len(images)} image(s) provided]\n\n{text_prompt}"
            
            messages.append({
                "role": "user",
                "content": full_prompt
            })
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 2000  # Increased limit for longer responses
                }
            }

            # Debug: Log payload structure (without full base64 images)
            debug_payload = {
                "model": payload["model"],
                "messages_count": len(payload["messages"]),
                "has_images": any("images" in msg for msg in payload["messages"])
            }
            if debug_payload["has_images"]:
                for i, msg in enumerate(payload["messages"]):
                    if "images" in msg:
                        debug_payload[f"message_{i}_image_count"] = len(msg["images"])
                        debug_payload[f"message_{i}_image_sizes"] = [len(img) for img in msg["images"]]
            
            logging.info(f"Sending to Ollama: {debug_payload}")

            # Retry logic for API calls
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.requests.post(
                        self.chat_api_url,
                        json=payload,
                        timeout=self.request_timeout
                    )
                    response.raise_for_status()
                    break  # Success, exit retry loop
                except Exception as e:
                    logging.warning(f"VLM API call failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        logging.error("All retries failed, returning empty response")
                        return ""
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Debug: log raw response
            logging.info(f"Ollama API Response Status: {response.status_code}")
            logging.info(f"Ollama API Response Headers: {response.headers}")
            
            result = response.json()
            logging.info(f"Ollama API Response JSON keys: {result.keys()}")
            logging.info(f"Message field: {result.get('message')}")
            
            # Extract response - handle both content and thinking fields
            message = result.get("message", {})
            output_text = message.get("content", "")
            
            # If content is empty but thinking exists, use thinking (for Qwen models)
            if not output_text and "thinking" in message:
                output_text = message["thinking"]
                logging.info(f"Using thinking field instead of content")
            
            logging.info(f"Extracted content length: {len(output_text)}")
            if len(output_text) == 0:
                logging.warning(f"Empty response! Full result: {result}")
            
            # Update history
            messages.append({
                "role": "assistant",
                "content": output_text
            })
            self.history = messages
            
            # Manage history length
            if history == 0:
                self.history = []
            elif len(self.history) > 2 * history + 1:  # +1 for system message
                # Keep system message and last 2*history messages
                if self.system_instruction:
                    self.history = [self.history[0]] + self.history[-2*history:]
                else:
                    self.history = self.history[-2*history:]

        except Exception as e:
            logging.error(f"OLLAMA CHAT API ERROR: {e}")
            return "OLLAMA API ERROR: Connection failed"

        return output_text

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Ollama model.
        Uses /api/generate endpoint for single-turn inference.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        # Build prompt
        full_prompt = ""
        if self.system_instruction:
            full_prompt += f"{self.system_instruction}\n\n"
        
        # Check if vision model
        is_vision_model = any(keyword in self.model.lower() for keyword in ['llava', 'bakllava', 'moondream', 'cogvlm'])
        
        if images and is_vision_model:
            # For vision models with /api/generate, we need to use base64 images in prompt
            # Note: /api/generate has limited vision support
            logging.warning("Using /api/generate with vision model. Consider using call_chat() instead.")
            full_prompt += text_prompt
        else:
            full_prompt += text_prompt
        
        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }

            response = self.requests.post(
                self.api_url,
                json=payload,
                timeout=self.request_timeout
            )
            response.raise_for_status()
            result = response.json()
            output_text = result.get("response", "")

        except Exception as e:
            logging.error(f"OLLAMA GENERATE API ERROR: {e}")
            return "OLLAMA API ERROR: Connection failed"
        except Exception as e:
            logging.error(f"OLLAMA GENERATE ERROR: {e}")
            return "OLLAMA API ERROR"

        return output_text

    def rewind(self):
        """Rewind the chat history by one step."""
        if len(self.history) > 1:
            # Remove last user-assistant pair
            if self.system_instruction and len(self.history) > 1:
                # Keep system message
                self.history = [self.history[0]] + self.history[:-2]
            else:
                self.history = self.history[:-2]

    def reset(self):
        """Reset the chat history."""
        self.history = []

    def get_spend(self):
        """Local model has no cost."""
        return 0


class DepthEstimator:
    """
    A class for depth estimation from images using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the depth estimation pipeline with the appropriate model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "Intel/zoedepth-kitti"
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)

    def call(self, image: np.array):
        """
        Perform depth estimation on an image.

        Parameters
        ----------
        image : np.array
            An RGB image for depth estimation.
        """
        original_shape = image.shape
        image_rgb = Image.fromarray(image[:, :, :3])
        depth_predictions = self.pipe(image_rgb)['predicted_depth']

        # Resize the depth map back to the original image dimensions
        depth_predictions = depth_predictions.squeeze().cpu().numpy()
        depth_predictions = cv2.resize(depth_predictions, (original_shape[1], original_shape[0]))

        return depth_predictions


class QwenVLClient(VLM):
    """
    VLM client for Alibaba Cloud's Qwen-VL (Tongyi Qianwen) online API.
    Uses DashScope SDK for API calls.
    """

    def __init__(self, model: str = "qwen-vl-max", api_key: str = None, 
                 max_image_res: int = 1024, system_instruction: str = None, **kwargs):
        """
        Initialize the Qwen-VL client.
        
        Parameters
        ----------
        model : str
            Model name (e.g., 'qwen-vl-max', 'qwen-vl-plus')
        api_key : str
            DashScope API key. If None, reads from DASHSCOPE_API_KEY env var.
        max_image_res : int
            Maximum image resolution (width or height)
        system_instruction : str
            System instruction for the model
        """
        super().__init__()
        
        try:
            import dashscope
            from dashscope import MultiModalConversation
        except ImportError:
            raise ImportError("Please install dashscope: pip install dashscope")
        
        self.model = model
        self.max_image_res = max_image_res
        self.system_instruction = system_instruction
        self.call_count = 0      # batch runner 计量: 成功调用次数 (episode 差分)

        # Set API key
        if api_key:
            dashscope.api_key = api_key
        else:
            import os
            # Try multiple environment variable names
            dashscope.api_key = os.getenv('QWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
            if not dashscope.api_key:
                raise ValueError("API key not provided. Set api_key parameter, QWEN_API_KEY, or DASHSCOPE_API_KEY environment variable.")
        
        self.client = MultiModalConversation()
        logging.info(f"Initialized QwenVLClient with model: {model}")

    def call_chat(self, history: int, images: list, text_prompt: str,
                  plain_text: bool = False) -> str:
        """
        Call Qwen-VL chat API with images and text prompt.

        Parameters
        ----------
        history : int
            Number of conversation history turns to include (currently not used for Qwen)
        images : list[np.array]
            List of RGB image arrays
        text_prompt : str
            Text prompt
        plain_text : bool
            True = 本次调用要自由文本 + 规定行 (FIRST_DIRECTION: /
            SCAN_SUSPICIOUS: 等六图扫描协议) → 不注入 JSON-only 系统指令。
            #26c 修复: 该参数此前只加在 OllamaVLM 上, 实弹用的是本类 →
            p26 三局 18 次关键分析全崩 (TypeError: unexpected keyword)。

        Returns
        -------
        str
            Model response text
        """
        try:
            # Encode images to base64
            image_contents = []
            for img in images:
                try:
                    base64_img = self._encode_image_to_base64(img)
                    image_contents.append({
                        "image": f"data:image/jpeg;base64,{base64_img}"
                    })
                except Exception as e:
                    logging.error(f"Error encoding image for Qwen: {e}")
            
            # Build messages
            messages = []
            
            # Add system message if provided
            # plain_text=True (#26c): 六图扫描协议要 FIRST_DIRECTION:/
            # SCAN_SUSPICIOUS: 规定行, JSON-only 系统指令会压制其输出
            # (p16 实弹三局 SCAN_ 行零解析的根因) — 跳过注入
            if self.system_instruction and not plain_text:
                messages.append({
                    "role": "system",
                    "content": [{"text": self.system_instruction}]
                })
            
            # Build user message with images and text
            user_content = image_contents + [{"text": text_prompt}]
            messages.append({
                "role": "user",
                "content": user_content
            })
            
            # Debug logging
            logging.info(f"Sending to Qwen-VL: model={self.model}, images={len(images)}, prompt_length={len(text_prompt)}")
            
            # Call API
            response = self.client.call(
                model=self.model,
                messages=messages,
                result_format='message'
            )
            
            # Check response status
            if response.status_code != 200:
                logging.error(f"Qwen API error: {response.code} - {response.message}")
                return ""
            
            # Extract response text
            output_text = response.output.choices[0].message.content[0]['text']
            logging.info(f"Qwen response received (length: {len(output_text)})")
            self.call_count += 1     # batch runner 计量

            return output_text
            
        except Exception as e:
            logging.error(f"Error calling Qwen-VL API: {e}", exc_info=True)
            return ""

    def get_spend(self):
        """成功调用次数 (batch runner 效率计量, 表 VIII)"""
        return self.call_count

    def _encode_image_to_base64(self, image: np.array) -> str:
        """
        Encode a numpy array image to base64 string.
        
        Parameters
        ----------
        image : np.array
            RGB image array
            
        Returns
        -------
        str
            Base64 encoded image string (without data:image prefix)
        """
        # Resize if needed
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim > self.max_image_res:
            scale = self.max_image_res / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str


class Segmentor:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the segmentation model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").to(self.device)

        # Get class ids for navigable regions
        id2label = self.model.config.id2label
        self.navigability_class_ids = [id for id, label in id2label.items() if 'floor' in label.lower() or 'rug' in label.lower()]

    def get_navigability_mask(self, im: np.array):
        """
        Generate a navigability mask from an input image.

        Parameters
        ----------
        im : np.array
            An RGB image for generating the navigability mask.
        """
        image = Image.fromarray(im[:, :, :3])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

        navigability_mask = np.isin(predicted_semantic_map, self.navigability_class_ids)
        return navigability_mask
