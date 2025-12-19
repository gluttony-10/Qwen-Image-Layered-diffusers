import os
import gc
from PIL import Image
import math
from mmgp import offload
import torch
import numpy as np
import gradio as gr
import socket
import psutil
import random
import argparse
import datetime
from diffusers import QwenImageLayeredPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration


parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument("--compile", action="store_true", help="是否启用compile加速")
parser.add_argument("--res_vram", type=int, default=2000, help="显存保留，单位MB。数值越大，占用显存越小，速度越慢")
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda" 
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，尝试使用FP32\033[0m')
        dtype = torch.float32
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"

os.makedirs("outputs", exist_ok=True)
repo_id = "./models/Qwen-Image-Layered"
budgets = int(torch.cuda.get_device_properties(0).total_memory/1048576 - args.res_vram)
stop_generation = False
mode_loaded = None
pipe = None
mmgp = None
lora_loaded = None
lora_loaded_weights = None
lora_dir = "models/lora"
if os.path.exists(lora_dir):
    lora_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
    lora_choices = sorted(lora_files)
else:
    lora_choices = []

def load_model(mode):
    global pipe, mmgp
    text_encoder = offload.fast_load_transformers_model(
        f"{repo_id}/text_encoder/mmgp.safetensors",
        do_quantize=False,
        modelClass=Qwen2_5_VLForConditionalGeneration,
        forcedConfigPath=f"{repo_id}/text_encoder/config.json",
    )
    if mode == "i2i":
        if pipe is not None:
            mmgp.release()
        transformer = offload.fast_load_transformers_model(
            f"{repo_id}/transformer/mmgp.safetensors",
            do_quantize=False,
            modelClass=QwenImageTransformer2DModel,
            forcedConfigPath=f"{repo_id}/transformer/config.json",
        )
        pipe = QwenImageLayeredPipeline.from_pretrained(
            repo_id, 
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=dtype,
            #low_cpu_mem_usage=False, 
        )
        pipe.set_progress_bar_config(disable=None)
    mmgp = offload.all(
        pipe, 
        pinnedMemory= ["transformer"],
        budgets={'*': budgets}, 
        extraModelsToQuantize = ["text_encoder"],
        compile=True if args.compile else False,
    )
    if torch.cuda.get_device_capability()[0] >= 8:
        pipe.transformer.set_attention_backend("flash")
    else:
        pipe.transformer.set_attention_backend("native")
    """offload.save_model(
        model=pipe.transformer, 
        file_path=f"{repo_id}/transformer/mmgp.safetensors", 
        config_file_path=f"{repo_id}/transformer/config.json",
    )
    offload.save_model(
        model=pipe.text_encoder, 
        file_path=f"{repo_id}/text_encoder/mmgp.safetensors", 
        config_file_path=f"{repo_id}/text_encoder/config.json",
    )"""

# 解决冲突端口（感谢licyk酱提供的代码~）
def find_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port


def stop_generate():
    global stop_generation
    stop_generation = True
    return "🛑 等待生成中止"


def generate_i2i(
    init_image,
    prompt,
    negative_prompt,
    cfg_normalize,
    use_en_prompt,
    true_cfg_scale,
    layers,
    num_inference_steps,
    batch_images,
    seed_param,
):
    global stop_generation, mode_loaded, lora_loaded, lora_loaded_weights
    if mode_loaded != "i2i":
        load_model("i2i")
        mode_loaded = "i2i"
    results = []
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt)
    negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(negative_prompt)
    for i in range(batch_images):
        if stop_generation:
            stop_generation = False
            yield results, f"✅ 生成已中止，最后种子数{seed+i-1}"
            break
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = pipe(
            image=init_image,
            #prompt=prompt,
            #negative_prompt=negative_prompt,
            cfg_normalize=cfg_normalize,
            use_en_prompt=use_en_prompt,
            true_cfg_scale=true_cfg_scale,
            layers=layers,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed+i),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        )
        output_image = output.images[0]
        for i, image in enumerate(output_image):
            image.save(f"outputs/{timestamp}{i:02d}.png")
            results.append(image)
        yield results, f"种子数{seed+i}，保存地址f'outputs/{timestamp}{i:02d}.png'"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    

with gr.Blocks(title="Qwen-Image-Layered-diffusers", theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans")])) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Qwen-Image-Layered-diffusers</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |Qwen-Image-Layered-diffusers
                <a href="https://github.com/gluttony-10/Qwen-Image-Layered-diffusers">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)
    with gr.Tabs():
        with gr.TabItem("图生图"):
            with gr.Row():
                with gr.Column():
                    image_i2i = gr.Image(label="输入图片(暂不支持纯文生，管线问题)", type="pil", height=300, image_mode="RGBA")
                    prompt_i2i = gr.Textbox(label="提示词", placeholder="请输入提示词...")
                    negative_prompt_i2i = gr.Textbox(label="负面提示词", placeholder="可留空")
                    generate_button_i2i = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                    with gr.Accordion("参数设置", open=True):
                        with gr.Row():
                            cfg_normalize_i2i = gr.Checkbox(label="cfg规范化", value=True)
                            use_en_prompt_i2i = gr.Checkbox(label="使用英文提示词", value=False)
                        with gr.Row():
                            true_cfg_scale_i2i = gr.Slider(label="True guidance scale（推荐4.0）", minimum=0, maximum=10, step=0.1, value=4.0)
                            layers_i2i = gr.Slider(label="图层数（推荐4）", minimum=1, maximum=10, step=1, value=4)
                        batch_images_i2i = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                        num_inference_steps_i2i = gr.Slider(label="采样步数（推荐50步）", minimum=1, maximum=100, step=1, value=28)
                        seed_param_i2i = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                with gr.Column():
                    info_i2i = gr.Textbox(label="提示信息", interactive=False)
                    image_output_i2i = gr.Gallery(label="生成结果", interactive=False)
                    stop_button_i2i = gr.Button("中止生成", variant="stop")
    # 图生图  
    gr.on(
        triggers=[generate_button_i2i.click, prompt_i2i.submit],
        fn = generate_i2i,
        inputs = [
            image_i2i,
            prompt_i2i,
            negative_prompt_i2i,
            cfg_normalize_i2i,
            use_en_prompt_i2i,
            true_cfg_scale_i2i,
            layers_i2i,
            num_inference_steps_i2i,
            batch_images_i2i,
            seed_param_i2i,
        ],
        outputs = [image_output_i2i, info_i2i]
    )
    stop_button_i2i.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_i2i]
    )


if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=find_port(args.server_port),
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )