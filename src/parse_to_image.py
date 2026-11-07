"""
PDF to image conversion utilities for NotesAssistant.
Converts PDF pages to individual image files.
"""

import os
import shutil

from pdf2image import convert_from_path

from settings import PDF_DPI


def pdf_to_images(pdf_path, output_path):
    """
    Converts all pages in a PDF file to individual image files.

    The output folder is emptied first. Otherwise pages left over from a
    previous, longer PDF would be embedded into the new index.

    Args:
        pdf_path (str): Path to the PDF file.
        output_path (str): Directory to save the output images.

    Returns:
        int: Number of pages/images generated.
    """
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    images = convert_from_path(pdf_path, dpi=PDF_DPI)
    for i, image in enumerate(images):
        image_filename = os.path.join(output_path, f'page_{i+1:03}.jpg')
        image.save(image_filename, 'JPEG')
    return len(images)
