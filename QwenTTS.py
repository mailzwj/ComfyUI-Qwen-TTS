import torchaudio
import os
import json
import dashscope
import requests
from io import BytesIO

USER_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

USER_CONFIG = json.load(open(USER_CFG_PATH, "r", encoding="utf-8"))

if not USER_CONFIG.get("bailian_api_key"):
    raise ValueError("Please set your Bailian API key in the config.json file.")

MENU_GROUP = "Qwen"

class QwenTTS:
    """
    QwenTTS Node for ComfyUI
    Reference: https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen-tts?modelGroup=qwen-tts
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id": (["qwen-tts-latest", "qwen-tts-2025-05-22", "qwen-tts-2025-04-10", "qwen-tts"], {
                    "default": USER_CONFIG.get("default_model", "qwen-tts-latest"),
                }),
                "content": ("STRING", {
                    "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "你好，千问！",  # Default text
                    "placeholder": "TTS text",
                }),
                "voice": (["Cherry", "Serena", "Ethan", "Chelsie", "Dylan", "Jada", "Sunny"], {
                    "default": "Sunny",  # Default voice
                })
            },
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("音频", "采样率")

    FUNCTION = "run"

    #OUTPUT_NODE = False

    CATEGORY = MENU_GROUP

    def run(self, model_id, content, voice):
        tts_res = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            # 仅支持qwen-tts系列模型，请勿使用除此之外的其他模型
            model=model_id,
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx"
            api_key=USER_CONFIG["bailian_api_key"],
            text=content,
            voice=voice,
        )
        response = requests.get(tts_res.output.audio["url"])
        response.raise_for_status()
        # 使用 BytesIO 读取音频数据
        # 创建内存文件对象
        audio_buffer = BytesIO(response.content)
        
        # 使用torchaudio加载
        waveform, sample_rate = torchaudio.load(audio_buffer)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, sample_rate)
    
class AudioInfo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", {}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("音频", "采样率", "时长")

    FUNCTION = "getInfo"

    #OUTPUT_NODE = False

    CATEGORY = MENU_GROUP

    def getInfo(self, audio):
        waveform = audio.get("waveform")
        sample_rate = audio.get("sample_rate")
        audioObj = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sample_rate)(waveform)
        duration = audioObj.size(2) / sample_rate
        return (audio, sample_rate, duration)

    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


# Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


# --- ComfyUI 节点映射 ---
NODE_CLASS_MAPPINGS = {
    "QwenTTS": QwenTTS,
    "AudioInfo": AudioInfo,
}

# ComfyUI 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTS": "千问TTS",
    "AudioInfo": "音频信息",
}
