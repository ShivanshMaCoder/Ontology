from fastapi import FastAPI, Query
from utils import FileManager, DataProcessor, GeminiClient, OntologyBuilder
import os
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel,Field
from typing import Optional

load_dotenv()
app = FastAPI()



class OntologyRequest(BaseModel):
    azure_connection_string: Optional[str] = Field(default_factory=lambda: os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    azure_golden_truth_path: Optional[str] = Field(default_factory=lambda: os.getenv('FILE_PATH'))
    azure_directory_path: Optional[str] = Field(default_factory=lambda: os.getenv('DIRECTORY_PATH'))

class TestOntologyRequest(BaseModel):
    azure_connection_string: Optional[str] = Field(default_factory=lambda: os.getenv('AZURE_STORAGE_CONNECTION_STRING'))
    azure_directory_path: Optional[str] = Field(default_factory=lambda: os.getenv('DIRECTORY_PATH'))




@app.post("/process_ontology/")
async def process_ontology(request:OntologyRequest):
    
    azure_connection_string = request.azure_connection_string
    azure_golden_truth_path = request.azure_golden_truth_path
    azure_directory_path = request.azure_directory_path
    azure_container = azure_directory_path.split('/')[3]
    processor = DataProcessor(azure_connection_string,
                              azure_container, azure_golden_truth_path)
    processor.clean_data()
    categories = {
        'competition': [], 'operations': [], 'internet': [],
        'accessories': [], 'phone': [], 'tv': [],
        'billing-support': [], 'general': [],
        'national-billing-issues': [], 'plans': [],
        'devices': [], 'features': [], 'programs': [],
        'channel-operations': [], 'system-and-tools': [],
        'account-support': [], 'promotions': []
    }
    processor.populate_categories(categories)
    processor.flatten_categories(categories)
    output_directory = 'HTML_FILES'
    file_manager = FileManager(
        output_directory, azure_connection_string, azure_directory_path, flag=False)
    unique_file_number_dict = file_manager.map_file_number_to_file_location()
    golden_truth_df = processor.change_source_files(unique_file_number_dict)
    genai_client = GeminiClient(api_key=os.getenv('GOOGLE_API_KEY'),flag=True)
    final_answer = dict()
    all_entities = []

    unique_categories = list(golden_truth_df['Category'].unique())

    for c in unique_categories:
        all_questions = []
        sources = list(
            golden_truth_df[golden_truth_df['Category'] == c]['Source'].unique())
        for source in sources:
            questions = list(golden_truth_df[(golden_truth_df['Category'] == c) & (
                golden_truth_df['Source'] == source)]['Question'])
            all_questions.extend(questions)
        entities = genai_client.gemini_call(
            genai_client.upload_multiple_files(sources), all_questions)
        all_entities.append(entities)

    final_answer = OntologyBuilder.build_ontology(final_answer, all_entities)
    try:
        file_path = OntologyBuilder.save_ontology(final_answer)
        print(file_path)
    except Exception as e:
        print(f'The error is {e}')
    return final_answer


@app.post("/sample_ontology/")
async def sample_ontology(request:TestOntologyRequest):
    azure_connection_string = request.azure_connection_string
    azure_directory_path = request.azure_directory_path
    output_directory = 'HTML_FILES'
    file_manager = FileManager(
        output_directory, azure_connection_string, azure_directory_path, flag=True)
    file_manager.download_blob()
    genai_client = GeminiClient(api_key=os.getenv('GOOGLE_API_KEY'),flag=False)
    final_answer = dict()
    all_entities = []
    all_questions=[]
    directory=output_directory
    source = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    entities = genai_client.gemini_call(
        genai_client.upload_multiple_files(source), all_questions)
    all_entities.append(entities)
 
    final_answer = OntologyBuilder.build_ontology(final_answer, all_entities)
    try:
        file_path = OntologyBuilder.save_ontology(final_answer)
        print(file_path)
    except Exception as e:
        print(f'The error is {e}')
    return final_answer


if __name__ == '__main__':
    uvicorn.run('source:app', host="0.0.0.0", port=8000)
