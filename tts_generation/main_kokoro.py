# Kokoro-82M: https://huggingface.co/hexgrad/Kokoro-82M
from audiomath import Sound
from pathlib import Path
from typing import List, Dict, Union
from kokoro import KPipeline
import soundfile as sf
import audiomath as am
from utils_tool import get_folder_path, timer
import numpy as np
import tempfile, os

'''
Language code reference:
'a'=American English, 'b'=British English, 'z'=Mandarin Chinese, 'j'=Japanese,
'e'=Spanish, 'f'=French, 'h'=Hindi, 'i'=Italian, 'p'=Brazilian Portuguese
'''

def _silence_wav(path: Path, duration_sec: float, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(duration_sec * sr), dtype=np.float32)
    sf.write(str(path), silence, sr)

def _safe_name(text: str) -> str:
    return text.replace(' ', '_')

def tts_offline(text: str, voice: str, output_file: Path, lang_code: str = 'a', speed: float = 1.0, sr: int = 24000):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code=lang_code)
    gen = pipe(text, voice=voice, speed=speed)

    audio_chunks = []
    for _, (_, _, audio) in enumerate(gen):
        audio_chunks.append(audio)
    audio_all = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    sf.write(str(output_file), audio_all, sr)

def _unit_wav_temp_path(tmp_dir: Path) -> Path:
    fd, name = tempfile.mkstemp(suffix=".wav", dir=str(tmp_dir))
    os.close(fd)
    return Path(name)

def ensure_tts_file_for_item(item: Dict, voice: str, tmp_dir: Path) -> Path:
    text = item['text']
    wav_path = _unit_wav_temp_path(tmp_dir)
    if text.startswith("silence"):
        _silence_wav(wav_path, duration_sec=1.0)
    else:
        tts_offline(text, voice, wav_path)
    return wav_path

def build_segments_with_silence(clips: List[Sound], start_times: List[float]):
    segments = []
    current_time = 0.0
    for s, st in zip(clips, start_times):
        if st > current_time:
            segments.append(st - current_time)   # silence seconds
            current_time = st
        segments.append(s)
        current_time += s.Duration()
    return segments

def make_tts_timeline(item_or_schedule: Union[Dict, List[Dict]], voice: str, out_wav: Path):
    schedule = [item_or_schedule] if isinstance(item_or_schedule, dict) else item_or_schedule

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)

        wav_paths = [ensure_tts_file_for_item(it, voice, tmp_dir) for it in schedule]

        clips = [Sound(str(p)) for p in wav_paths]
        start_times = [float(it['start']) for it in schedule]

        segments = build_segments_with_silence(clips, start_times)
        final = am.Concatenate(*segments)
        final.Write(str(out_wav))
        
@timer
def main():
    source = 'kokoro'
    output_path = Path(get_folder_path('./data', source))

    # If text starts with "silence", a 1-second silence file will be created on the fly (no cache).
    start_time = 0.1
    text_list = [
        {'text': 'banker 0 points',        'start': start_time},
        {'text': 'banker 1 points',        'start': start_time},
        {'text': 'banker 2 points',        'start': start_time},
        {'text': 'banker 3 points',        'start': start_time},
        {'text': 'banker 4 points',        'start': start_time},
        {'text': 'banker 5 points',        'start': start_time},
        {'text': 'banker 6 points',        'start': start_time},
        {'text': 'banker 7 points',        'start': start_time},
        {'text': 'banker 8 points',        'start': start_time},
        {'text': 'banker 9 points',        'start': start_time},
        {'text': 'player 0 points',        'start': start_time},
        {'text': 'player 1 points',        'start': start_time},
        {'text': 'player 2 points',        'start': start_time},
        {'text': 'player 3 points',        'start': start_time},
        {'text': 'player 4 points',        'start': start_time},
        {'text': 'player 5 points',        'start': start_time},
        {'text': 'player 6 points',        'start': start_time},
        {'text': 'player 7 points',        'start': start_time},
        {'text': 'player 8 points',        'start': start_time},
        {'text': 'player 9 points',        'start': start_time},
        {'text': 'please place your bets', 'start': start_time},
        {'text': 'no more bets',           'start': start_time},
        {'text': 'open cards',             'start': start_time},
        {'text': 'good luck',             'start': start_time},
        {'text': 'silence',                'start': start_time},
    ]

    voice_list = [
        # 'af_heart',
        # 'af_alloy',
        # 'af_aoede',
        'af_bella',  # Recommend
        # 'af_jessica',
        # 'af_kore',
        # 'af_nicole',
        # 'af_nova',
        # 'af_river',
        # 'af_sarah',
        # 'af_sky',
    ]

    for text in text_list:
        for voice in voice_list:
            text_combination = _safe_name(text['text'])
            output_file = output_path / f"tts_{text_combination}.wav"
            make_tts_timeline(text, voice, Path(output_file))

if __name__ == '__main__':
    main()
