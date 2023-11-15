from flask import Flask, request, jsonify, redirect
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
import io
from PIL import Image
from pydantic import BaseModel
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = image_path
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

class ImageCaption(BaseModel):
    caption: str

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

# @app.post("/predict/", response_model=ImageCaption)
# def predict(file: UploadFile = File(...)):
#     # Load the image file into memory
#     contents = file.file.read()
#     image = Image.open(io.BytesIO(contents))
#     result = predict_step([image])
#     print(result)
#     x = JSONResponse(content={"caption": result})
#     return x

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Load the image file into memory
        contents = file.read()
        image = Image.open(io.BytesIO(contents))
        print(image)
        result = predict_step([image])
        print(result)
        response_data = {"caption": result}
        return jsonify(response_data)

    return "Error", 400

# Redirect the user to the documentation
# @app.get("/", include_in_schema=False)
# def index():
#     return RedirectResponse(url="/docs")



if __name__ == "__main__":
    app.run(debug=True,port = 8000)