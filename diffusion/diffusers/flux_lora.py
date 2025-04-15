# import torch
# from diffusers import FluxPipeline

# # 初始化前清空缓存
# torch.cuda.empty_cache()

# pipe = FluxPipeline.from_pretrained(
#     "/home/jiuxia/ftp_file/FLUX.1-dev",
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     use_safetensors=True
# )

# # 启用所有优化措施
# pipe.enable_sequential_cpu_offload()
# pipe.enable_attention_slicing(1)
# pipe.enable_vae_slicing()

# print(pipe.scheduler.compatibles)
# #pipe.load_lora_weights("dark_fantasy", weight_name="dark_fantasy_lora.safetensors", adapter_name="dark_fantasy")
# pipe.unet.load_lora_adapter("dark_fantasy", weight_name="dark_fantasy_lora.safetensors", adapter_name="dark_fantasy")
# prompt = "A dog holding a sign that says hello world"
# image = pipe(
#     prompt,
#     height=256,   # 降低分辨率
#     width=256,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=128,  # 缩短序列长度
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-dev-lora.png")


import torch
from diffusers import FluxPipeline

# 初始化管道
pipe = FluxPipeline.from_pretrained(
    "/home/jiuxia/ftp_file/FLUX.1-dev",
    torch_dtype=torch.float16
)

# 关键步骤：安装PEFT后需要显式启用
pipe.enable_lora()
print(pipe.transformer.config)
# 正确加载LoRA权重
#import pdb;pdb.set_trace()
pipe.load_lora_weights("dark_fantasy", weight_name="dark_fantasy_lora.safetensors")
# pipe.load_lora_weights(
#     "dark_fantasy",
#     weight_name="dark_fantasy_lora.safetensors",
#     #adapter_name="dark_fantasy",
#     adapter_kwargs={
#         "adapter_type": "text_to_image",
#         # Flux模型正确的维度获取方式
#         "cross_attention_dim": 8,
#         "attention_head_dim": 64
#     }
# )



# 推理时需要指定使用哪个适配器
prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    adapter_names=["dark_fantasy"],  # 激活指定LoRA
    height=128,
    width=128
).images[0]
image.save("flux-lora-output.png")
