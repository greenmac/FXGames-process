import torch
from lipsync import LipSync
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

def lipsync_and_replace_audio_segment(
    src_video_path: str,
    src_audio_path: str,
    model_weights: str,
    lip_video_path: str,
    final_output_path: str,
    replace_start: float,
    replace_end: float,
    device: str = 'cpu'
):
    # 使用 weights_only=False 明確允許完整 unpickle
    checkpoint = torch.load(model_weights, map_location=device, weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)

    lip = LipSync(
        model='wav2lip',
        checkpoint_path=None,
        nosmooth=True,
        device=device,
        cache_dir='cache_frames',
        save_cache=True
    )
    # 無內部 model 屬性，故改用 checkpoint_path 方式操作
    # 此時我們假設 LipSync 支援直接使用預載的 state dict:
    try:
        lip.model.load_state_dict(state, strict=False)
        lip.model.eval()
    except AttributeError:
        # 若 LipSync 未開放 model 屬性，回退為傳統方式
        lip = LipSync(
            model='wav2lip',
            checkpoint_path=model_weights,
            nosmooth=True,
            device=device,
            cache_dir='cache_frames',
            save_cache=True
        )

    lip.sync(src_video_path, src_audio_path, lip_video_path)

    lip_video = VideoFileClip(lip_video_path)
    b_audio = AudioFileClip(src_audio_path)

    pre = lip_video.subclip(0, replace_start)
    mid = lip_video.subclip(replace_start, replace_end).with_audio(
        b_audio.subclip(replace_start, replace_end)
    )
    post = lip_video.subclip(replace_end, lip_video.duration)

    final = concatenate_videoclips([pre, mid, post])
    final.write_videofile(final_output_path, codec='libx264', audio_codec='aac')

if __name__ == "__main__":
    root = './data/Klingai'
    A = f'{root}/ori/Dealing_cards_01.mp4'
    B = f'{root}/adj/Lip_Sync_Game_starting.mp4'
    weights = 'weights/wav2lip_gan.pth'
    temp = f'{root}/temp_lip.mp4'
    out = f'{root}/final.mp4'
    start, end = 0.5, 3

    lipsync_and_replace_audio_segment(
        src_video_path=A,
        src_audio_path=B,
        model_weights=weights,
        lip_video_path=temp,
        final_output_path=out,
        replace_start=start,
        replace_end=end,
        device='cpu'
    )
