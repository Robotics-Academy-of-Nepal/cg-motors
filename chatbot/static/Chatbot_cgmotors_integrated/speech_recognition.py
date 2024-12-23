from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import os

def transcribe_audio(file_path):
    """
    Transcribe audio using Azure Speech Services.
    """
    # Load credentials
    load_dotenv()
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    
    # Configure speech recognizer
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "ne-NP"]
    )
    audio_config = speechsdk.audio.AudioConfig(filename=file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_detect_source_language_config,
        audio_config=audio_config
    )

    # Perform transcription
    result = speech_recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    else:
        raise Exception("Speech recognition failed.")
