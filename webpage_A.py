# import required packages/libraries
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import io
from huggingface_hub import InferenceClient
import os


# Fetch API key from environment variable
api_key = os.getenv("HF_API_KEY")

client = InferenceClient(
    provider="hf-inference",
    api_key=api_key
)

# create the page title
st.set_page_config(page_title="AI Product Customisation A", layout="wide",
                   menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })


# title and description
st.title("AI-Driven Merchandise Customisation")
st.write("Not satisfied with the base product you want? Simply follow the steps below and try customising it to your liking!")
st.divider()

## split into 2 columns, with some padding on the right #################################################
left, right = st.columns([1, 1])

### left column ###
#-------------------------------------------------------------------------------------------------------
# image uploading
with left:
    image_url = st.text_input("Enter an image url")
    uploaded_image = st.file_uploader("Or upload an image of a product you want to customise", 
                                      type=["png", "jpg", "jpeg"])
    
    # function to fetch image url
    def load_image_url(url):
        try: 
            response = requests.get(url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            else:
                st.error("❌ Failed to load image. Please check the URL.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
        return None
    
    
    # display the uploaded image
    if image_url:   # option to upload via image's url
        img = load_image_url(image_url)
        if img:     
            st.image(img, caption="Image ready", use_container_width=False)
        
    elif uploaded_image:    # option to upload via image file
        img = Image.open(uploaded_image)
        st.image(img, caption="Image ready", use_container_width=False)

### right column ###
#-------------------------------------------------------------------------------------------------------
# text prompt input & custom image generation
with right:
    text_prompt = st.text_area("Enter your customisation prompt.", 
                               placeholder="e.g., Add a rainbow pattern to this T-shirt",
                               height=160,
                               max_chars=250)
    
    
    # functional button to generate customised image via API call
    gen_Image_button = st.button("Generate Customised Image", key='button1')
    placeholder = st.empty()
    if gen_Image_button:
        if text_prompt and (uploaded_image or image_url):
            placeholder.text("⚙️ AI model will process this soon...")
            
            # determine the image to use (either file/url)
            if uploaded_image:
                img = Image.open(uploaded_image)
            else:
                img = load_image_url(image_url)
            
            # to generate the customised image via api call to hugging face model
            if img:
                # converts image upload into temporary url so model can read
                if uploaded_image is not None:
                    # Save the file temporarily
                    image_url = "temp_image.jpg"
                    with open(image_url, "wb") as f:
                        f.write(uploaded_image.getbuffer())


                image_bytes = client.image_to_image(
                    image=image_url,
                    prompt=text_prompt,
                    model="lllyasviel/sd-controlnet-canny",
                    )


                if image_bytes:
                    st.image(image_bytes, caption="Custom Product Image")
                else:
                    st.error("❌ Failed to generate customised image.")
            
            else:
                st.error("❌ Invalid image input.")
        else:
            placeholder.text("Please ensure both image and text prompt have been input.")
    

##########################################################################################################

st.write(" # ")     # line breaks
st.divider()
st.write("#### Please leave a feedback on whether you like the custom product 😀 so we can improve!")


bottomleft, bottomright = st.columns([1, 1])

# Feedback rating (slider from 0 to 5 stars)
with bottomleft:
    rating = st.slider("Rate the generated image:", 1, 5 ,4)
    
# User feedback response       
with bottomright:
    if rating:
        st.text_area("Comment your feedback here:",
                            placeholder="What improvements would you like to see?",
                            height=140,
                            max_chars=250)
