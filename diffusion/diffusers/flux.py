import torch
from diffusers import FluxPipeline

# 初始化前清空缓存
torch.cuda.empty_cache()

pipe = FluxPipeline.from_pretrained(
    "/home/jiuxia/ftp_file/FLUX.1-dev",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True
)

# 启用所有优化措施
pipe.enable_sequential_cpu_offload()
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()

print(pipe.scheduler.compatibles)

prompt = "A dog holding a sign that says hello world"
image = pipe(
    prompt,
    height=256,   # 降低分辨率
    width=256,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=128,  # 缩短序列长度
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")

