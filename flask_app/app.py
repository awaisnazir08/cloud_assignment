import io
import logging
import torch
from flask import Flask, request, send_file, render_template
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to check GPU availability
def check_gpu_availability():
    return torch.cuda.is_available()

# Global variable to store the pipeline (will remain None if no GPU)
pipe = None

# Try to load pipeline, but only if GPU is available
if check_gpu_availability():
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", 
            torch_dtype=torch.float16,  # Use half precision for GPU
            variant="fp16"
        ).to("cuda")
        
        # Disable safety checker
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        
        # GPU-specific optimizations
        pipe.enable_attention_slicing()
    except Exception as e:
        logger.error(f"Error loading GPU pipeline: {e}")
        pipe = None
else:
    logger.warning("No GPU available. Pipeline cannot be loaded.")

def run_inference(prompt):
    # Recheck GPU availability before inference
    if not check_gpu_availability() or pipe is None:
        raise RuntimeError("No GPU available for inference")
    
    try:
        image = pipe(
            prompt, 
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        
        img_data = io.BytesIO()
        image.save(img_data, "PNG")
        img_data.seek(0)
        return img_data
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise

@app.route('/', methods=['GET', 'POST'])
def myapp():
    # Render the initial page
    if request.method == 'GET':
        # Check GPU availability
        if not check_gpu_availability() or pipe is None:
            return render_template('index.html', error="GPU Not Available")
        return render_template('index.html')
    
    # Handle form submission
    if request.method == 'POST':
        # Check GPU availability
        if not check_gpu_availability() or pipe is None:
            return render_template('index.html', error="GPU Not Available")
        
        prompt = request.form.get('prompt', '')
        
        # Prompt validation
        if not prompt:
            return render_template('index.html', error="Please enter a prompt")
        
        if len(prompt) > 200:
            return render_template('index.html', error="Prompt too long (max 200 characters)")
        
        try:
            img_data = run_inference(prompt)
            return send_file(img_data, mimetype='image/png')
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            return render_template('index.html', error="An error occurred during image generation")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
