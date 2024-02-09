'''Code generated via ChatGPT'''

from PIL import Image
import os
import tqdm as tqdm

def downscale_images(input_folder, output_folder, scale_factor):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for image_file in tqdm.tqdm(image_files):
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Open the image
        with Image.open(input_path) as img:
            # Calculate new dimensions
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)

            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

            # Save the resized image
            resized_img.save(output_path)

if __name__ == "__main__":
    # Set your input and output folders here
    input_folder = "./abstract-art/"
    output_folder = "../../../../output"
    
    # Set the scale factor (e.g., 0.5 for 50% reduction)
    scale_factor = 0.25

    # Call the function to downscale images
    downscale_images(input_folder, output_folder, scale_factor)

    print("Image downscaling complete.")

