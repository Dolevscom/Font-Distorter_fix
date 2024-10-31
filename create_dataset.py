import os
import json
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import streamlit as st

# Paths
fonts_dir = "/Users/dolevsmac/Desktop/FontDistorter/FontDistorter/fonts"
output_file = "google_fonts_extended_dataset.json"
google_fonts_file = "google-fonts.json"  # Path to Google Fonts JSON with category data

specimen_dir = "specimens"  # Folder to store specimen images
os.makedirs(specimen_dir, exist_ok=True)

# Helper function to classify the font characteristics
def classify_font(ttfont):
    characteristics = {}
    
    # Extract basic information
    characteristics["family"] = ttfont["name"].getName(1, 3, 1, 1033).toStr() if ttfont["name"].getName(1, 3, 1, 1033) else "Unknown"
    
    # Extract weight
    characteristics["weight"] = ttfont["OS/2"].usWeightClass if "OS/2" in ttfont else "Unknown"
    
    # Extract italic/slant style
    characteristics["italic"] = bool(ttfont["head"].macStyle & 2) if "head" in ttfont else False
    
    # Extract width
    width_class = ttfont["OS/2"].usWidthClass if "OS/2" in ttfont else 5  # Normal width as default
    width_map = {
        1: "Ultra-condensed",
        2: "Extra-condensed",
        3: "Condensed",
        4: "Semi-condensed",
        5: "Normal",
        6: "Semi-expanded",
        7: "Expanded",
        8: "Extra-expanded",
        9: "Ultra-expanded"
    }
    characteristics["width"] = width_map.get(width_class, "Unknown")

    # Classify style heuristically
    # characteristics["style"] = "Serif" if "serif" in characteristics["family"].lower() else "Sans-serif"
    
    # Calculate line height (if available)
    if "OS/2" in ttfont:
        characteristics["line_height"] = (
            ttfont["OS/2"].sTypoAscender - ttfont["OS/2"].sTypoDescender + ttfont["OS/2"].sTypoLineGap
        )
    else:
        characteristics["line_height"] = "Unknown"

    # Calculate x-height
    try:
        x_height = ttfont["OS/2"].sxHeight
        characteristics["x_height"] = x_height
    except AttributeError:
        characteristics["x_height"] = "Unknown"
    
    # Estimate font contrast
    if "glyf" in ttfont and ttfont["glyf"].glyphs.get("I"):
        glyph = ttfont["glyf"]["I"]
        if hasattr(glyph, "coordinates"):
            vertical_strokes = [coord[1] for coord in glyph.coordinates]
            contrast_ratio = (max(vertical_strokes) - min(vertical_strokes)) / characteristics["line_height"]
            characteristics["contrast"] = "High" if contrast_ratio > 0.5 else "Low"
        else:
            characteristics["contrast"] = "Unknown"
    else:
        characteristics["contrast"] = "Unknown"

    # Set curvature based on heuristic (if 'O' has circular shapes or not)
    if "glyf" in ttfont and "O" in ttfont["glyf"].glyphs:
        glyph = ttfont["glyf"]["O"]
        if hasattr(glyph, "coordinates") and len(glyph.coordinates) > 10:  # Approximation
            characteristics["curvature"] = "Round"
        else:
            characteristics["curvature"] = "Angular"
    else:
        characteristics["curvature"] = "Unknown"

    return characteristics

# Helper function to create a specimen image for the font
def create_specimen(font_path, font_family):
    try:
        specimen_text = "Sample Text"
        font = ImageFont.truetype(font_path, 40)
        image = Image.new("RGB", (300, 100), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.text((10, 25), specimen_text, font=font, fill=(0, 0, 0))
        
        specimen_path = os.path.join(specimen_dir, f"{font_family}.png")
        image.save(specimen_path)
        return specimen_path
    except Exception as e:
        print(f"Could not create specimen for font {font_family}: {e}")
        return None

# Function to create dataset
def create_json():
    font_data = []
    for root, dirs, files in os.walk(fonts_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".otf"):
                font_path = os.path.join(root, file)
                try:
                    # Load the font file and classify
                    ttfont = TTFont(font_path)
                    characteristics = classify_font(ttfont)
                    characteristics["file_path"] = font_path  # Path to font file
                    
                    # Generate and save specimen path
                    specimen_path = create_specimen(font_path, characteristics["family"])
                    characteristics["specimen_path"] = specimen_path if specimen_path else "N/A"
                    
                    font_data.append(characteristics)
                except Exception as e:
                    print(f"Could not process font {file}: {e}")

    font_data = sorted(font_data, key=lambda x: x["family"])
    # Save dataset with specimen paths
    with open(output_file, "w") as f:
        json.dump(font_data, f, indent=4)
    
    print(f"Font dataset with specimens saved to {output_file}")

# Run JSON creation function
# create_json()

# # Load JSON data into a DataFrame
# Function to add category information to the existing JSON
def adding_category():
    # Load the existing dataset
    with open(output_file, "r") as f:
        font_data = json.load(f)

    # Load the Google Fonts JSON file with category information
    with open(google_fonts_file, "r") as f:
        google_fonts_data = json.load(f)

    # Create a dictionary mapping from font family to category
    category_mapping = {font_family: details.get("category", "Unknown") for font_family, details in google_fonts_data.items()}

    # Add the category to each entry in the dataset
    for entry in font_data:
        font_family = entry["family"]
        entry["category"] = category_mapping.get(font_family, "Unknown")

    # Overwrite the existing dataset with updated category information
    with open(output_file, "w") as f:
        json.dump(font_data, f, indent=4)

    print("Added category information to the existing dataset.")

# Run the function to add categories
# adding_category()
# Function to replace 'style' with 'category' information in the existing JSON
def update_category_field():
    # Load the existing dataset
    with open(output_file, "r") as f:
        font_data = json.load(f)

    # Load the Google Fonts JSON file with category information
    with open(google_fonts_file, "r") as f:
        google_fonts_data = json.load(f)

    # Create a dictionary mapping from font family to category
    category_mapping = {font_family: details.get("category", "Unknown") for font_family, details in google_fonts_data.items()}

    # Replace 'style' with 'category' for each entry in the dataset
    for entry in font_data:
        font_family = entry["family"]
        entry["category"] = category_mapping.get(font_family, "Unknown")  # Add the category field
        entry.pop("style", None)  # Remove the 'style' field if it exists

    # Overwrite the existing dataset with updated information
    with open(output_file, "w") as f:
        json.dump(font_data, f, indent=4)

    print("Replaced 'style' with 'category' in the existing dataset.")

# Run the function to update the JSON
update_category_field()

# Load the modified JSON data into a DataFrame
# df = pd.read_json(output_file)
# st.write("Font Dataset with Updated Category Field", df)