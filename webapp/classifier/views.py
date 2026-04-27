import base64

from django.shortcuts import render

from .model_utils import get_predictor


def home(request):
    context = {}

    if request.method == "POST":
        uploaded = request.FILES.get("image")

        if not uploaded:
            context["error"] = "Please choose an image file."
            return render(request, "classifier/index.html", context)

        try:
            predictor = get_predictor()
            prediction = predictor.predict(uploaded, top_k=5)

            uploaded.seek(0)
            encoded = base64.b64encode(uploaded.read()).decode("ascii")
            mime_type = uploaded.content_type or "image/jpeg"

            context.update(
                {
                    "preview_src": f"data:{mime_type};base64,{encoded}",
                    "top_prediction": prediction["top1"],
                    "topk_predictions": prediction["topk"],
                }
            )
        except Exception as exc:
            context["error"] = f"Prediction failed: {exc}"

    return render(request, "classifier/index.html", context)
