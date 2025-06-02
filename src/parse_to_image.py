from pdf2image import convert_from_path
import os

def toimages(pdf_path):
    output_path = './output_images'
    os.makedirs(output_path,exist_ok=True)
    images = convert_from_path(pdf_path, dpi=200,first_page=1,last_page=5)

    for i, image in enumerate(images):
        image_filename = os.path.join(output_path, f'page_{i+1:03}.png')
        image.save(image_filename, 'PNG')