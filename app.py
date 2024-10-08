import streamlit as st
from PIL import Image
import requests
import io
from transformers import StableDiffusionControlNetPipeline, ControlNetModel

# Set up Streamlit app
st.title("Architectural Sketch Renderer")

# Text input for the prompt
prompt = st.text_input("Enter a prompt for the rendering")

# Upload button for the sketch/massing
uploaded_file = st.file_uploader("Upload your sketch or massing", type=["png", "jpg", "jpeg"])

if uploaded_file is not None and prompt:
    # Open the uploaded image
    input_image = Image.open(uploaded_file)
    
    # Resize image to the model's expected size
    input_image = input_image.resize((512, 512))
    
    # Convert image to byte array
    byte_array = io.BytesIO()
    input_image.save(byte_array, format='PNG')
    byte_array = byte_array.getvalue()

    # Display the uploaded sketch
    st.image(input_image, caption="Uploaded Sketch", use_column_width=True)

    # Initialize the ControlNet model and pipeline
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
    
    # Generate image based on the prompt and sketch
    generated_image = pipe(prompt=prompt, image=input_image, height=1024, width=1920).images[0]
    
    # Display the generated image
    st.image(generated_image, caption="Generated Image", use_column_width=True)

    # Download button for the generated image
    buf = io.BytesIO()
    generated_image.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download image",
        data=byte_im,
        file_name="generated_image.png",
        mime="image/png"
    )
