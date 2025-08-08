from huggingface_hub import hf_hub_download

# 下載 wav2lip.pth
path = hf_hub_download(
    repo_id="numz/wav2lip_studio",
    filename="Wav2lip/wav2lip.pth",
    repo_type="model"
)
print("下載完成，檔案位置：", path)
