# https://huggingface.co/hexgrad/Kokoro-82M
from kokoro import KPipeline
import soundfile as sf
from utils_tool import get_folder_path
from utils_tool import timer

source = 'kokoro'

def tts_offline(text: str, voice, output_file: str):
    '''
    'a' = American English  
    'b' = British English  
    'z' = Mandarin Chinese  
    'j' = Japanese  
    'e' = Spanish  
    'f' = French  
    'h' = Hindi  
    'i' = Italian  
    'p' = Brazilian Portuguese
    '''
    pipe = KPipeline(lang_code='a')

    gen = pipe(text, voice=voice, speed=1.0)

    for _, (_, _, audio) in enumerate(gen):
        sf.write(output_file, audio, 24000)

@timer
def main():
    output_path = get_folder_path('./data', source)
    
    text_list = [
        # 'Banker 0 points',
        # 'Banker 1 points',
        # 'Banker 2 points',
        # 'Banker 3 points',
        # 'Banker 4 points',
        # 'Banker 5 points',
        # 'Banker 6 points',
        # 'Banker 7 points',
        # 'Banker 8 points',
        # 'Banker 9 points',
        # 'Player 0 points',
        # 'Player 1 points',
        # 'Player 2 points',
        # 'Player 3 points',
        # 'Player 4 points',
        # 'Player 5 points',
        # 'Player 6 points',
        # 'Player 7 points',
        # 'Player 8 points',
        # 'Player 9 points',
        # 'Please place your bets',
        # 'No more bets',
        # 'Banker wins',
        # 'Player wins',
        'Good luck'
    ]

    voice_list = [
        # 'af_heart',
        # 'af_alloy',
        # 'af_aoede',
        'af_bella', # Recommend
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
            text_combination = text.title().replace(' ', '')
            output_file = f"{output_path}/tts_{text_combination}.wav"
            tts_offline(text, voice, output_file)
        
if __name__ == "__main__":
    main()
