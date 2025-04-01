# import required packages/libraries
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO


############################################################################
#########           HOW TO LAUNCH & TEST THE WEBPAGE           #############
# 1. Change directory and launch of webpage
#   Type  `cd Desktop\DSA3101` in your terminal
#   to navigate to the cloned repo locally.
#   If not cloned yet, follow the README instructions on Github.
#   Then type `streamlit run webpage_A.py` in your terminal to launch webpage
# 2. Input image (URL or upload)
#   After the webpage launches,
#   test using this URL below/just use any random image u have:
#   https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg
# 3. Customisation text
#   Enter some random text for the customisation prompt, then
#   click the 'Generate Customised Image' button. 
#   The text below the button differs if there are missing required inputs.
############################################################################
############################################################################



## set Hugging Face API
MODEL1 = "lllyasviel/sd-controlnet-canny"    
MODEL2 = "stabilityai/stable-diffusion-xl-refiner-1.0"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL2}"
TOKEN = ""
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


# function to call Hugging Face Model via API 
def generate_custom_image(prompt, image):
    # convert to Base64 image
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # API request payload
    payload = {
        "image": encoded_image,  # Send image in Base64 format
        "inputs": prompt
    }
    # Make the request
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))  # Convert response bytes to an image
    else:
        st.error(f"❌ API Error {response.status_code}: {response.text}")
        return None

    

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
            
            # to generate the customised image via huging face model
            # ----- uncomment the code below only when ready to test, since API call is limited ------------
            
            if img:
                image_result = generate_custom_image(text_prompt, img)
                if image_result:
                    st.image(image_result, caption="Custom Product Image")
                else:
                    st.error("❌ Failed to generate customized image.")    
            
            # ----- uncomment the code above only when ready to test, since API call is limited ------------
            else:
                st.error("❌ Invalid image input.")
        else:
            placeholder.text("[Please ensure both image and text prompt have been input.]")
    

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






