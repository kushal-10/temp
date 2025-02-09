from pydantic import BaseModel, Field
from typing import List 
from restack_ai.workflow import workflow 
import pymupdf
import requests

class PdfWorkflowInput(BaseModel):
    file_upload: List[dict] = Field(files=True) 

@workflow.defn()
class PdfWorkflow: 
    @workflow.run
    async def run(self, input: PdfWorkflowInput):
        file = input.file_upload[0]
        print(file['name'])

        response = requests.get(f"{'http://localhost:6233'}/api/download/{file['name']}")
        response.raise_for_status()  # Raise an error for bad responses
        content = response.content
        
        doc = pymupdf.Document(stream=content)
        
        pdfContent = ""
        for page in doc:
            text = page.get_text()
            pdfContent += text
        
         
        return { "content": pdfContent } 


        

        