"""
PDF to image conversion utilities for NotesAssistant.
Converts PDF pages to individual image files.
"""

from pdf2image import convert_from_path
import os
import PyPDF2

def pdf_to_images(pdf_path, output_path):
    """
    Converts all pages in a PDF file to individual image files.

    Args:
        pdf_path (str): Path to the PDF file.
        output_path (str): Directory to save the output images.

    Returns:
        int: Number of pages/images generated.
    """
    with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pdflen = len(reader.pages)
    os.makedirs(output_path,exist_ok=True)
    images = convert_from_path(pdf_path, dpi=200,first_page=1,last_page=pdflen)
    pdflen = 0
    for i, image in enumerate(images):
        image_filename = os.path.join(output_path, f'page_{i+1:03}.jpg')
        pdflen+=1
        image.save(image_filename, 'JPEG')
    return pdflen