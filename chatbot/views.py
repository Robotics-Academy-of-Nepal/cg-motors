from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from chatbot.static.Chatbot_cgmotors_integrated.speech_recognition import transcribe_audio
from chatbot.static.Chatbot_cgmotors_integrated.chatbot_inference import get_chatbot_response
import tempfile
import os
from django.conf import settings


@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            # Get uploaded audio file
            uploaded_file = request.FILES.get("audio_file")

            if not uploaded_file:
                return JsonResponse({"error": "No audio file provided."}, status=400)

            # Create a temporary file to save the uploaded audio file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Transcribe audio
            transcribed_text = transcribe_audio(temp_file_path)

            # Get chatbot response and intent
            chatbot_result = get_chatbot_response(transcribed_text)
            response_text = chatbot_result["response"]
            intent_tag = chatbot_result["intent"]
            audio_path = chatbot_result["audio_path"]

            # Move audio file to the media directory
            media_path = os.path.join(settings.MEDIA_ROOT, "audio.wav")
            os.rename(audio_path, media_path)
            
            audio_url = request.build_absolute_uri(settings.MEDIA_URL + "audio.wav")


            # Return JSON response
            return JsonResponse({
                "intent_tag": intent_tag,
                "response": response_text,
                "audio_url": audio_url
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)


