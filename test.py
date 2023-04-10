import speech_recognition as sr
import pyttsx3
from pyllamacpp.model import Model

r = sr.Recognizer()
r.dynamic_energy_threshold = False
r.energy_threshold=1000
mic = sr.Microphone(device_index=0)

# pyttsx3 Set-up
engine = pyttsx3.init()
# engine.setProperty('rate', 180) #200 is the default speed, this makes it slower
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # 0 for male, 1 for female

# Takes user voice input from microphone and returns the audio.
# If there's no audio, it will return an empty array
def listen_for_voice(timeout:int|None=5):
    with mic as source:
            print("\n Listening...")
            r.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = r.listen(source, timeout)
            except:
                return []
    print("no longer listening")
    return audio

def generate_voice(response):
            engine.say(f"{response}")
            engine.runAndWait()

while True:
    audio = listen_for_voice(timeout=None)
    try:
        user_input = r.recognize_google(audio)
    except Exception as e:
        print(e) 
        continue
    print(user_input + "\n")
    model = Model(ggml_model='./models/gpt4all-converted.bin', n_ctx=512)
    generated_text = model.generate(user_input, n_predict=258)
    print(generated_text)
    generate_voice(generated_text)