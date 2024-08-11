import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")
Eleven = config["Eleven_key_AUDIO"]

def translator(audio_file):
    
    #1: Transcribir el audio a texto

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="Spanish", fp16=False)
        transcription = result["text"]
    except Exception as e:
        raise gr.Error(
            f"An error occurred transcribing the audio file: {str(e)}")
    
    
    #2: Traducir el texto a inglés  

    try:
        en_transcription = Translator(from_lang="es", to_lang="en").translate(transcription)
    except Exception as e:
        raise gr.Error(
            f"An error occurred translating the transcription: {str(e)}")
    
    print(f"texto orignial {en_transcription}")
    #3: Generar audio traducido

    try:
        client = ElevenLabs(api_key=Eleven)

        response = client.text_to_speech.convert(
            voice_id="flq6f7yk4E4fJM5XTYuZ",  # Michael
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=en_transcription,
            model_id="eleven_turbo_v2",  # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        save_file_path = "audios/translated_audio.mp3"

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise gr.Error(f"An error occurred generating the translated audio: {str(e)}")

    return save_file_path

web = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=[
        gr.Audio(label="Ingles")
        #En este espacio pueden añadir mas idiomas
        ],
    title="Speech to Text Translator",
    description="Translate your speech to text"
)

web.launch(share=True)