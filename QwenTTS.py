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

os.environ["DASHSCOPE_API_KEY"] = USER_CONFIG["bailian_api_key"]

class QwenTTS:
    """
    QwenTTS Node for ComfyUI
    Reference: https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen-tts?modelGroup=qwen-tts

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tuple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    DEPRECATED (`bool`):
        Indicates whether the node is deprecated. Deprecated nodes are hidden by default in the UI, but remain
        functional in existing workflows that use them.
    EXPERIMENTAL (`bool`):
        Indicates whether the node is experimental. Experimental nodes are marked as such in the UI and may be subject to
        significant changes or removal in future versions. Use with caution in production workflows.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "model_id": (["qwen-tts-latest", "qwen-tts-2025-05-22", "qwen-tts-2025-04-10", "qwen-tts"], {
                    "default": USER_CONFIG.get("default_model", "qwen-tts-latest"),
                    "title": "模型"
                }),
                "content": ("STRING", {
                    "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "你好，千问！",  # Default text
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

    CATEGORY = "Qwen"

    # def check_lazy_status(self, image, string_field, int_field, float_field, print_to_screen):
    #     """
    #         Return a list of input names that need to be evaluated.

    #         This function will be called if there are any lazy inputs which have not yet been
    #         evaluated. As long as you return at least one field which has not yet been evaluated
    #         (and more exist), this function will be called again once the value of the requested
    #         field is available.

    #         Any evaluated inputs will be passed as arguments to this function. Any unevaluated
    #         inputs will have the value None.
    #     """
    #     if print_to_screen == "enable":
    #         return ["int_field", "float_field", "string_field"]
    #     else:
    #         return []

    def run(self, model_id, content, voice):
        tts_res = dashscope.audio.qwen_tts.SpeechSynthesizer.call(
            # 仅支持qwen-tts系列模型，请勿使用除此之外的其他模型
            model=model_id,
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx"
            api_key=os.environ["DASHSCOPE_API_KEY"],
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

    # """
    #     The node will always be re executed if any of the inputs change but
    #     this method can be used to force the node to execute again even when the inputs don't change.
    #     You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
    #     executed, if it is different the node will be executed again.
    #     This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
    #     changes between executions the LoadImage node is executed again.
    # """
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
    "QwenTTS": QwenTTS
}

# ComfyUI 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTS": "千问TTS"
}
