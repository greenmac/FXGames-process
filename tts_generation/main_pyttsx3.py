import pyttsx3
from utils_tool import get_folder_path

source = 'pyttsx3'

def tts_offline(text: str, output_file: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)     # 語速，預設約 200
    engine.setProperty('volume', 1.0)   # 音量範圍 0.0–1.0
    engine.save_to_file(text, output_file)
    engine.runAndWait()
    print(f"Speech generation is complete: {output_file}")

if __name__ == "__main__":
    text = "Please place your bets."
    output_path = get_folder_path('./data', source)
    output_file = f"{output_path}/tts_output_{source}.wav"
    
    tts_offline(text, output_file)

