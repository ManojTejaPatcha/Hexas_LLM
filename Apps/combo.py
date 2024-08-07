from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import io
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import pyflakes.api
import pyflakes.reporter
import io as sysio

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(question, image=None):
    if image is not None:
        response = model_vision.generate_content([question, image])
    else:
        response = model_text.generate_content(f"{question}. Please respond in English.")
    return response.text

def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    image = pipe(prompt, guidance_scale=7.5).images[0]
    return image

def debug_code(code):
    output_buffer = sysio.StringIO()
    reporter = pyflakes.reporter.Reporter(output_buffer, output_buffer)
    pyflakes.api.check(code, 'provided_code', reporter=reporter)
    output = output_buffer.getvalue()
    if not output:
        return "Debugging Output: No errors found."
    else:
        return f"Debugging Output:\n{output}"

st.set_page_config(page_title="Hexas Application")

st.header("Hexas Application")

model_text = genai.GenerativeModel("gemini-pro")
model_vision = genai.GenerativeModel("gemini-1.5-flash")

option = st.radio("Choose option:", ("Text-based prompt", "Upload image"))

if option == "Text-based prompt":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    form = st.form(key='chat_form')
    input_message = form.text_input(label='You:', key='input')
    generate_image_button = form.checkbox('Generate Image')
    debug_code_button = form.checkbox('Debug Code')

    if form.form_submit_button(label='Send') and input_message:
        if generate_image_button:
            image = generate_image(input_message)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.session_state.chat_history.append({"question": input_message, "image": byte_im})
        elif debug_code_button:
            debug_output = debug_code(input_message)
            st.session_state.chat_history.append({"question": input_message, "response": debug_output})
        else:
            response = get_gemini_response(input_message)
            st.session_state.chat_history.append({"question": input_message, "response": response})

    if st.session_state.chat_history:
        with st.expander("Chat History"):
            for chat in st.session_state.chat_history:
                st.write(f"You: {chat['question']}")
                if 'response' in chat:
                    st.write(f"Gemini: {chat['response']}")
                    if "" in chat['response']:
                        st.code(chat['response'].strip(""), language='python')
                if 'image' in chat:
                    st.image(chat['image'], caption="Generated Image")
                st.write("---")

    if st.session_state.chat_history:
        st.subheader("Current Chat")
        last_chat = st.session_state.chat_history[-1]
        st.write(f"You: {last_chat['question']}")
        if 'response' in last_chat:
            st.write(f"Gemini: {last_chat['response']}")
            if "" in last_chat['response']:
                st.code(last_chat['response'].strip(""), language='python')
        if 'image' in last_chat:
            st.image(last_chat['image'], caption="Generated Image")
        st.write("---")

elif option == "Upload image":
    st.header("Hexas LLM Application")
    input_text = st.text_input("Input Prompt ", key="input")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Tell me about the image")

    if submit:
        response = get_gemini_response(input_text, image)
        st.subheader("The Response is")
        st.write(response)

st.write("""
<style>
.stTextInput > label {
    display: none;
}
</style>
""", unsafe_allow_html=True)
