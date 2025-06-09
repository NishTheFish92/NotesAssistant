from pdf2image import convert_from_path
import os
import PyPDF2

def toimages(pdf_path):
    #converts all the pages in the PDF to individual images
    pdflen = 0
    output_path = './output_images'
    with open('../PDFs/Mod5.pdf', "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pdflen = len(reader.pages)
    os.makedirs(output_path,exist_ok=True)
    images = convert_from_path(pdf_path, dpi=200,first_page=1,last_page=pdflen)

    for i, image in enumerate(images):
        image_filename = os.path.join(output_path, f'page_{i+1:03}.png')
        image.save(image_filename, 'PNG')