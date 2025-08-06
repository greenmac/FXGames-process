import pyttsx3

def tts_offline(text: str, filename: str):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)     # 語速，預設約 200
    engine.setProperty('volume', 1.0)   # 音量範圍 0.0–1.0
    engine.save_to_file(text, filename)
    engine.runAndWait()

if __name__ == "__main__":
    text = "Place your bets."
    tts_offline(text, "./data/tts_output.wav")
    print("語音生成完成：./data/tts_output.wav")
