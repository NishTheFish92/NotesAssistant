from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
import base64
from tenacity import retry, stop_after_attempt, wait_exponential
import os

@retry(
    stop=stop_after_attempt(3),  # Try 3 times
    wait= wait_exponential(multiplier=1, min=4, max=10),  # Wait between 4-10 seconds, increasing exponentially
)
def img_to_llm(image_file_path,llm):
    #Passes an image to the LLM and then returns a description of the image
    try:
        with open(image_file_path, "rb") as image_file:
            image_data = image_file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        output_parser = StrOutputParser()

        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe the image I am sending. Send it as plain text in paragraphs with no formatting"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                }
            ]
        )

        response = llm.invoke([message])
        return response.content
        
    except Exception as e:
        print(f"Attempt failed: {str(e)}")
        raise
def img2txt(output_path,n,llm):
    #Sends the description of the images into a text file
    for i in range(n):
        llm_output = ""
        image_filename = os.path.join(output_path,f'page_{i+1:03}.png')
        llm_output = img_to_llm(image_filename,llm)
        text_filename = os.path.join(output_path, f'page_{i+1:03}.txt')
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(llm_output)