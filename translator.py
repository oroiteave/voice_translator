import gradio as gr
import whisper as ws
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")
ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

def translator(audio_file):

    #Transcribe texto
    try:
        model = ws.load_model("base")
        result = model.transcribe(audio_file, language = "Spanish",fp16 = False)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error transcribiendo un texto: {str(e)}")

    print(f"Texto original: {transcription}")
            
    #Traductor de texto
    try:
        translator = Translator(from_lang="es", to_lang="en")
        en_transcription = translator.translate(transcription)
    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error traducioendo el texto: {str(e)}")    

    print(f"Texto traducido: {en_transcription}")
    #Generador de audio
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    response = client.text_to_speech.convert(
        voice_id="bIHbv24MWmeRgasZH58o", # Will
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=en_transcription,
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    save_file_path = "audios/en.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
                
    return save_file_path

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath"
    ),
    outputs=[gr.Audio(label="Ingles")],
    title="Traductor de voz",
    description="Traductor de voz con IA a varios idiomas"
)

web.launch()