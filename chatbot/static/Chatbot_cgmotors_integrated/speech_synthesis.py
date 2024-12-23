from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import os
import re

def detect_language(input_text):
    nepali_regex = re.compile(r'[\u0900-\u097F]')
    if nepali_regex.search(input_text):
        return 'ne-NP'  # Nepali language code
    return 'en-US'      # Default to English language code



def synthesize_speech(text, output_path="media/audio.wav"):
    """
    Synthesize text into speech using Azure Speech Services and save it to the output path.

    Args:
        text (str): The text to convert to speech.
        output_path (str): Path where the synthesized audio will be saved.

    Returns:
        str: The file path of the synthesized audio.

    Raises:
        Exception: If speech synthesis fails or credentials are missing.
    """
    load_dotenv()
    # Retrieve Azure Speech Service credentials
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")

    # Check if credentials are available
    if not speech_key or not speech_region:
        raise Exception("Azure Speech Service credentials (SPEECH_KEY and SPEECH_REGION) are not set.")

    # Configure Azure speech synthesizer
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Detect language and set the appropriate voice
    detected_language = detect_language(text)
    if detected_language == 'en-US':
        speech_config.speech_synthesis_voice_name = 'en-US-JennyNeural'  # English voice
    elif detected_language == 'ne-NP':
        speech_config.speech_synthesis_voice_name = 'ne-NP-HemkalaNeural'  # Nepali voice
    else:
        raise Exception(f"Unsupported language detected: {detected_language}")

    # Perform speech synthesis
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"[DEBUG] Speech synthesis completed. Audio saved to: {output_path}")
        return output_path
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        raise Exception(f"Speech synthesis canceled: {cancellation_details.reason}. Error details: {cancellation_details.error_details}")
    else:
        raise Exception("Speech synthesis failed for an unknown reason.")


