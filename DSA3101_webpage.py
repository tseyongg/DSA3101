# import required packages/libraries
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

############################################################################
#########           HOW TO LAUNCH & TEST THE WEBPAGE           #############
# 1.
#   type  `cd Desktop\DSA3101` in your terminal
#   to navigate to the cloned repo locally, if not cloned yet,
#   follow the README instructions on Github
#   then type `streamlit run .\DSA3101_webpage.py` in your terminal to launch
# 2.
#   after the webpage launches,
#   test using this URL below/just use any random image u have:
#   https://images-na.ssl-images-amazon.com/images/I/41SyGjt4KdL.jpg
# 3.
#   enter some random text for the customisation prompt, then
#   click the 'Generate Customised Image' button, 
#   the text below the button differs if u r missing any input image/prompt
############################################################################
############################################################################

# create the page title
st.set_page_config(page_title="AI Product Customisation", layout="wide",
                   menu_items={
        'Get Help': 'https://www.youtube.com',
        'Report a bug': "https://www.google.com",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

# title and description
st.title("AI-Driven Merchandise Customisation")
st.write("Not satisfied with the base product you want? Simply follow the steps below and try customising it to your liking!")

## split into 2 columns, with some padding on the right #################################################
left, right, right_pad = st.columns([10, 9, 1])

### left column ###
#-------------------------------------------------------------------------------------------------------
# image/image url uploading
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
    
    if image_url:
        img = load_image_url(image_url)
        if img:     
            st.image(img, caption="Image ready", use_container_width=False)
        
    elif uploaded_image:
        st.image(image_url, caption="Image ready", use_container_width=False)

### right column ###
#-------------------------------------------------------------------------------------------------------
# text prompt input
with right:
    text_prompt = st.text_area("Enter your customisation prompt.", 
                               placeholder="e.g., Add a rainbow pattern to this T-shirt",
                               height=350,
                               max_chars=500)
    

##########################################################################################################

# functional button to generate customised image
gen_Image_button = st.button("Generate Customised Image", key='button1')
placeholder = st.empty()
if gen_Image_button:
    if text_prompt and (uploaded_image or image_url):
        placeholder.text("⚙️ AI model will process this soon...")

    else:
        placeholder.text("❌ Failed to generate image. Please try again later.")
else:
    placeholder.text("[Please input your image and/or text prompts.]")







