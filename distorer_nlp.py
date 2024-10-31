import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageTk,UnidentifiedImageError
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog, Label
import openai
import os
import cv2
import numpy as np
import streamlit as st
from io import BytesIO



# Initialize OpenAI API key
openai.api_key = " ->>>>my api key"

# Set the path for tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
google_fonts_directory = "/Users/dolevsmac/Desktop/FontDistorter/FontDistorter/fonts"
img_path = None

def load_image(image_path):
    img_color = Image.open(image_path).convert("RGB")  # Keep the color intact
    img_gray = img_color.convert("L")
    return img_color, img_gray

def extract_text_and_boxes(img_gray, min_box_height=10):
    """
    Extracts text and bounding boxes from the image, filtering small boxes.

    Parameters:
    - img_gray: Grayscale image to extract text from.
    - min_box_height: Minimum height of text boxes to filter out artifacts.

    Returns:
    - text: Extracted text as a string.
    - boxes: Filtered bounding boxes as a list of strings.
    """
    text = pytesseract.image_to_string(img_gray)
    print(f"Extracted text: {text}")
    
    all_boxes = pytesseract.image_to_boxes(img_gray)
    img_height = img_gray.size[1]
    
    # Filter out small boxes
    filtered_boxes = []
    for box in all_boxes.splitlines():
        b = box.split(' ')
        top, bottom = int(b[4]), int(b[2])
        box_height = img_height - bottom - (img_height - top)
        if box_height >= min_box_height:
            filtered_boxes.append(box)
    
    return text, '\n'.join(filtered_boxes)

def inpaint_content_aware_fill(image, boxes, padding=10):
    """
    Inpaints text areas with dynamic padding based on the background complexity.

    Parameters:
    - image: PIL Image to process.
    - boxes: Bounding boxes of text.
    - initial_padding: Base padding around text to cover surrounding areas.
    """
    try:
        # Load the uploaded file as an image
        image = Image.open(image)
        img_np = np.array(image.convert("RGB"))
    except UnidentifiedImageError:
        st.write("Error: The uploaded file is not a valid image.")
        return None

    img_height, img_width = img_np.shape[:2]
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw rectangles over the areas identified for inpainting
    for box in boxes.splitlines():
        b = box.split(' ')
        left, bottom, right, top = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        adjusted_top = img_height - top
        adjusted_bottom = img_height - bottom
        cv2.rectangle(mask, 
                        (left - padding, adjusted_top - padding), 
                        (right + padding, adjusted_bottom + padding), 
                        255, -1)

    # Perform inpainting
    inpainted_img = cv2.inpaint(img_np, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted_img)

def calculate_median_y(boxes, img_height):
    y_positions = [img_height - int(box.split()[4]) for box in boxes.splitlines()]
    if y_positions:
        median_y = sum(y_positions) // len(y_positions)
    else:
        median_y = img_height // 2
    return median_y

def identify_font(img_gray):
    pipe = pipeline("image-classification", model="gaborcselle/font-identifier")
    results = pipe(img_gray)
    top_font = results[0]['label']
    print(f"Top predicted font: {top_font}")
    return top_font

def get_alternative_fonts(top_font, text_extracted):
    """
    Fetches a list of alternative fonts that are visually distinct from the provided top_font.

    Parameters:
    - top_font: The name of the font to get alternatives for.

    Returns:
    - A list of alternative font names.
    """

    prompt = (
            f"Give me the 5 fonts that are the most different from {top_font}, "
            f"focusing on characteristics like weight, style, and personality. "
            f"refferin also to the context of the text extracted from the image, {text_extracted}."
            f"Return them as an array of font names, ensuring they are included in Google Fonts."
        )

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        # Extract the response text
        processed_text = response.choices[0].message.content
        print(f"ChatGPT response:\n{processed_text}")

        # Assume it's a bullet list and extract font names by splitting lines
        fonts_list = [line.split('"')[1] for line in processed_text.splitlines() if '"' in line]
        return fonts_list
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def find_first_valid_font(fonts_list, root_dir):
    """
    Finds the first valid font file path from a list of font names.

    Parameters:
    - fonts_list: List of font names to check.
    - root_dir: Root directory of Google Fonts.

    Returns:
    - The path to the first valid font file, or None if not found.
    """
    for font_name in fonts_list:
        font_path = find_font_file(font_name, root_dir)
        if font_path:
            print(f"Found valid font: {font_name} at {font_path}")
            return font_path
    print("No valid font path found in the provided font list.")
    return None

def find_font_file(font_name, root_dir):
    normalized_font_name = font_name.lower().replace(" ", "")
    for license_dir in ['ofl', 'apache', 'ufl']:
        font_dir = os.path.join(root_dir, license_dir)
        if os.path.isdir(font_dir):
            for sub_dir in os.listdir(font_dir):
                sub_dir_path = os.path.join(font_dir, sub_dir)
                if os.path.isdir(sub_dir_path) and normalized_font_name in sub_dir_path.lower():
                    for file in os.listdir(sub_dir_path):
                        if file.endswith(".ttf"):
                            return os.path.join(sub_dir_path, file)
    return None

def apply_new_font(img_color, boxes, font_path, median_y, padding=5):
    img_draw = img_color.copy()
    draw = ImageDraw.Draw(img_draw)
    
    if font_path and os.path.isfile(font_path):
        font_size = 40
        font = ImageFont.truetype(font_path, font_size)

        for box in boxes.splitlines():
            b = box.split(' ')
            left, bottom, right, top = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            char = b[0]
            adjusted_top = img_color.height - top

            # Only overlay text directly on top of the original text positions
            draw.text((left, adjusted_top), char, font=font, fill=(0, 0, 0))
        
    return img_draw

def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        img = Image.open(img_path)
        img.thumbnail((400, 400))  # Resize for display
        img_display = ImageTk.PhotoImage(img)
        uploaded_image_label.config(image=img_display)
        uploaded_image_label.image = img_display

def process_image():
    global img_path
    if img_path:
        img_color, img_gray = load_image(img_path)
        text, boxes = extract_text_and_boxes(img_gray)
        
        # Inpaint to remove text
        img_color = inpaint_content_aware_fill(img_path, boxes)
        
        # Font replacement
        median_y = calculate_median_y(boxes, img_color.height)
        top_font = identify_font(img_gray)
        alternative_fonts = get_alternative_fonts(top_font)
        
        # Find the first valid font path in the list of alternatives
        font_path = find_first_valid_font(alternative_fonts, google_fonts_directory)
        
        # Apply new font if a valid font path is found
        if font_path:
            processed_img = apply_new_font(img_color, boxes, font_path, median_y)
            
            # Display the processed image
            processed_img.thumbnail((400, 400))
            processed_img = ImageTk.PhotoImage(processed_img)
            result_image_label.config(image=processed_img)
            result_image_label.image = processed_img
        else:
            st.write("No valid font path found. Please check your font directory.")
    else:
        st.write("No image path found. Please upload an image first.")



# Streamlit GUI
st.title("Font Distorter")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image in the left column
    col1, col2 = st.columns(2)
    with col1:
        img_color = Image.open(uploaded_file)
        img_color.thumbnail((400, 400))
        st.image(img_color, caption="Uploaded Image", use_column_width=True)

    # Processing with a spinner
    with st.spinner("Processing image..."):
        img_color, img_gray = load_image(uploaded_file)

        # Extract text and bounding boxes
        text, boxes = extract_text_and_boxes(img_gray)
        st.write("Extracted text:", text)

        # Inpaint to remove text
        img_color = inpaint_content_aware_fill(uploaded_file, boxes)
        if img_color is None:
            st.write("Image processing failed. Please upload a valid image.")
        else:
            # Font replacement
            median_y = calculate_median_y(boxes, img_color.height)
            top_font = identify_font(img_gray)
            alternative_fonts = get_alternative_fonts(top_font,text)
            
            # Find the first valid font path
            font_path = find_first_valid_font(alternative_fonts, google_fonts_directory)

            # Apply new font if path is valid and display in the right column
            with col2:
                if font_path:
                    processed_img = apply_new_font(img_color, boxes, font_path, median_y)
                    st.image(processed_img, caption="Processed Image", use_column_width=True)
                else:
                    st.write("Font path does not exist or was not found.")
    
        # Success message once processing is complete
        st.success("Image processing complete!")


# os.system("streamlit run distorer_nlp.py")
