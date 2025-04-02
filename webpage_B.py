# import required packages/libraries
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO
import io


############################################################################
#########           HOW TO LAUNCH & TEST THE WEBPAGE           #############
# 1. Change directory and launch of webpage
#   Type  `cd Desktop\DSA3101` in your terminal
#   to navigate to the cloned repo locally.
#   If not cloned yet, follow the README instructions on Github.
#   Then type `streamlit run webpage_B.py` in your terminal to launch webpage
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
# MODEL1 = "lllyasviel/sd-controlnet-canny"    
# MODEL2 = "stabilityai/stable-diffusion-xl-refiner-1.0"
# API_URL = f"https://api-inference.huggingface.co/models/{MODEL1}"
# TOKEN = ""
# HEADERS = {"Authorization": f"Bearer {TOKEN}"}



from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key="",
)

# # output is a PIL.Image object
# image = client.image_to_image(
#     "cat.png",
#     prompt="Turn the cat into a tiger.",
#     model="lllyasviel/sd-controlnet-canny",
# )




# API_URL = "https://router.huggingface.co/hf-inference/models/lllyasviel/sd-controlnet-canny"
# headers = {"Authorization": "Bearer hf_xxx"}

# def query(payload):
#     with open(payload["inputs"], "rb") as f:
#         img = f.read()
#         payload["inputs"] = base64.b64encode(img).decode("utf-8")
#     response = requests.post(API_URL, headers=headers, json=payload)
#     return response.content




## encode image to base64
def encode_image(image):
    """Convert PIL Image to Base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # Save image to a buffer
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# image_bytes = query({
#     "inputs": "cat.png",
#     "parameters": {
#         "prompt": "Turn the cat into a tiger."
#     }
# })

# You can access the image with PIL.Image for example
# import io
# from PIL import Image
# image = Image.open(io.BytesIO(image_bytes))





# ## function to call Hugging Face Model via API 
# def generate_custom_image(prompt, image):
#     # convert to Base64 image
#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
#     # API request payload
#     payload = {
#         "image": encoded_image,  # Send image in Base64 format
#         "inputs": prompt
#     }
#     # Make the request
#     response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=120)
#     if response.status_code == 200:
#         return Image.open(BytesIO(response.content))  # Convert response bytes to an image
#     else:
#         st.error(f"❌ API Error {response.status_code}: {response.text}")
#         return None



# create the page title
st.set_page_config(page_title="AI Product Customisation B", layout="wide",
                   menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })


st.divider()
st.title("AI-Driven Merchandise Customisation")
# st.write("Not satisfied with the base product you want? Simply follow the steps below and try customising it to your taste!")
st.markdown('''
    :red[Not satisfied] with the base product? Simply follow the steps below and :green[try customising] it to your :rainbow[taste!]''')

# step 1: choose your item source
st.subheader("Step 1: Choose Image Source")
step1 = st.selectbox("Select your image source", ["Select", "Image link", "Upload image"])

# step 2: based on selection, upload image source
image_link = None
image_upload = None

if step1 == "Image link":
    st.subheader("Step 2: Insert Image Link")
    image_link = st.text_input("Enter the URL of the image")
elif step1 == "Upload image":
    st.subheader("Step 2: Uplaod an Image")
    image_upload = st.file_uploader("Choose an image file", type = ["jpg", "jpeg", "png"])

# function to fetch image url
def load_image_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error("Failed to load image. Please ensure that it is a valid URL.")
    except Exception as e:
        st.error(f"Error: {e}")
    return None

if image_link:
    img = load_image_url(image_link) # output is img
    if img:
        st.image(img, caption = "Image ready", use_container_width = False)


elif image_upload:
    img = Image.open(image_upload) # output is img
    st.image(img, caption = "Image ready", use_container_width=False)



# step 3: Customisation details - only shown if step 2 has been filled
if step1 != "Select" and (image_link or image_upload):
    st.subheader("Step 3: Describe the Customisation")
    customisation_text = st.text_area("What kind of customisations do you want?")

    # step 4: Customise button
    if st.button("Customise"):
        if not customisation_text:
            st.error("Please mention what kind of customisations you want.")
        else:
            # proceed with customisation
            st.success("Your customisation is in progress!")
            st.write(f"Image Source: {step1}")
            if image_link:
                st.write(f"Image Link: {image_link}")
            if image_upload:
                st.image(image_upload, caption="Uploaded Image", use_column_width=True)

            st.write(f"Customisation Details: {customisation_text}")

            if img:                
                
                # if image_upload: # convert into bytes for function to read
                #     image_link = encode_image(img)

                image_bytes = client.image_to_image(
                    image=image_link,
                    prompt=customisation_text,
                    model="lllyasviel/sd-controlnet-canny",
                    )


                if image_bytes:
                    st.image(image_bytes, caption="Custom Product Image")
                else:
                    st.error("❌ Failed to generate customized image.")


else:
    if step1 != "Select":
        st.warning("Please provide an image before proceeding to customisation.")







## feedback portion

# st.write(" # ")     # line break
st.divider()
st.write("#### Are you satisfied with your customised product? 😀")

# Feedback buttons (Thumbs Up / Thumbs Down)
col1, col2, col3, padding = st.columns([1, 1, 1, 6]) # more space on the right

with col1:
    thumbs_up = st.button("👍 Good", key="thumbs_up")
with col2:
    neutral = st.button(" 👌 Okay", key="neutral")
with col3:
    thumbs_down = st.button("👎 Bad", key="thumbs_down")

# User feedback response
bottomleft, bottomright = st.columns([1, 1])

with bottomleft:
    if thumbs_up:
        st.success("✅ Thank you for your positive feedback!")
    elif neutral or thumbs_down:
        st.error("❌ We'll work on improving the customization!")
        
with bottomright:
    if thumbs_up or neutral or thumbs_down:
        st.text_area("Comment your feedback here:",
                            placeholder="What improvements would you like to see?",
                            height=140,
                            max_chars=250)
