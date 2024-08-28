import os
import pandas as pd
import numpy as np
import google.generativeai as genai
import json
import logging
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobClient
from tqdm import trange, tqdm
from typing import List
import random
import shutil

logging.basicConfig(level=logging.INFO)


class FileManager:
    def __init__(self, directory, azure_connection_string, azure_directory_path, flag):
        self.directory = directory
        self.azure_connection_string = azure_connection_string
        self.azure_directory_path = azure_directory_path
        self.Flag = flag

    def delete_directory(self):
        """Deletes the directory and all its contents."""
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
            print(f"Directory {self.directory} and all its contents have been deleted.")
        else:
            print(f"Directory {self.directory} does not exist.")

    def list_all_blobs_in_folder(self, container_client, folder_name: str):
        """Lists all blobs in a given folder within an Azure blob container."""
        try:
            return list(container_client.list_blobs(name_starts_with=folder_name))
        except Exception as e:
            print(f"Error listing blobs in folder {folder_name}: {e}")
            return []

    def download_blob(self) -> List[str]:
        """Downloads files from Azure directory, selecting a subset if there are many.

        Args:
            azure_connection_string (str): Connection String for Azure.
            azure_directory_path (str): Path to the directory on Azure.
            local_dir (str): Path to the local directory for downloads.

        Returns:
            list: List of downloaded file paths.
        """

        try:
            # Extract container and folder names
            container_name = self.azure_directory_path.split("/")[3]
            folder_name = self.azure_directory_path.split(
                container_name + "/")[-1]
            if len(folder_name) != 0 and folder_name[0] == "/":
                folder_name = folder_name[1:]

            # Connect to Azure Blob Storage
            blob_service_client = BlobServiceClient.from_connection_string(
                self.azure_connection_string)
            container_client = blob_service_client.get_container_client(
                container_name)

            # List blobs in the specified folder
            blobs = self.list_all_blobs_in_folder(
                container_client, folder_name)
            if not blobs:
                print(
                    f"No blobs found in directory: {self.azure_directory_path}")
                return []
            files_to_download = []
            if self.Flag:
                if len(blobs) <= 10:
                    files_to_download = blobs
                else:
                    files_to_download = random.sample(blobs, 10)
            else:
                files_to_download = blobs

            # Download selected blobs
            downloaded_files = []
            for blob in tqdm(files_to_download, desc='Downloading Blobs'):
                if blob.name.endswith('/'):  # Skip directories
                    continue

                # Extract relative path of blob inside directory
                relative_path = blob.name[len(folder_name):].lstrip('/')

                # Create local directory if it doesn't exist
                local_file_path = os.path.join(self.directory, relative_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                # Download the blob to a local file
                blob_client = container_client.get_blob_client(blob.name)
                try:
                    with open(local_file_path, "wb") as download_file:
                        download_file.write(
                            blob_client.download_blob().readall())
                    downloaded_files.append(local_file_path)
                except Exception as e:
                    tqdm.write(f"Error downloading blob {blob.name}: {e}")

            return downloaded_files
        except Exception as e:
            print(f"An error occurred while downloading blobs: {e}")
            return []

    def map_file_number_to_file_location(self):
        mp = {}
        all_files = self.download_blob()

        for s in all_files:
            ch = s.split(".")[-3]
            if ch not in mp:
                mp[ch] = s
            else:
                logging.warning(f'This id: {ch} is repeating')
        return mp


class DataProcessor:
    def __init__(self, connection_string, container, blob):
        self.connection_string = connection_string
        self.container = container
        self.blob = blob
        blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=self.container, blob=self.blob)
        blob = blob_client.download_blob().content_as_bytes()
        self.df = pd.read_excel(blob)

    def clean_data(self):
        try:
            self.df.dropna(subset=['Source', 'Questions'], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            logging.info("Data cleaned successfully")
        except Exception as e:
            logging.error(f'Error cleaning data: {e}')
            raise e

    def populate_categories(self, mp):
        try:
            for i in range(self.df.shape[0]):
                for k, v in mp.items():
                    if k in self.df['Source'][i]:
                        mp[k].append(
                            [self.df['Source'][i], self.df['Questions'][i]])
        except Exception as e:
            logging.error(f'Error populating categories: {e}')
            raise e

        try:
            for k, v in list(mp.items()):
                if len(v) < 1:
                    del mp[k]
        except Exception as e:
            logging.error(f'Error cleaning categories: {e}')
            raise e

    def change_source_files(self, mp):
        for x in range(self.df.shape[0]):
            data = self.df.loc[x, 'Source'].split("/")[-1].split(".")[0]
            if data in mp:
                self.df.loc[x, 'Source'] = mp[data]
            else:
                logging.warning(f'id: {data} not present in html directory')
                self.df.loc[x, 'Source'] = np.nan

        self.df.dropna(subset=['Source'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def flatten_categories(self, mp):
        try:
            logging.info("Starting to flatten categories.")

            category = []
            source = []
            question = []

            for k, v in mp.items():
                for x in v:
                    category.append(k)
                    source.append(x[0])
                    question.append(x[1])

            columns = ['Category', 'Source', 'Question']
            df = pd.DataFrame(
                list(zip(category, source, question)), columns=columns)

            self.df = df

            logging.info("Categories flattened successfully.")
            logging.debug(f"Flattened DataFrame head:\n{self.df.head()}")

        except Exception as e:
            logging.error(f"Error flattening categories: {e}")
            raise e


class GeminiClient:
    def __init__(self, api_key,flag):
        genai.configure(api_key=api_key)
        self.Flag=flag

    def upload_multiple_files(self, file_paths):
        uploaded_files = []
        for path in file_paths:
            try:
                uploaded_file = genai.upload_file(path=path)
                uploaded_files.append(uploaded_file)
                logging.info(
                    f"Uploaded file '{uploaded_file.display_name}' as: {uploaded_file.uri}")
            except Exception as e:
                logging.error(f"Error uploading file '{path}': {e}")
        return uploaded_files

    def gemini_call(self, uploaded_files, questions):
        try:
            if self.Flag:
                sep_ques = "\n".join(questions)
                model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest",
                                            generation_config={"response_mime_type": "application/json"})
                with open('prompts/ontology_prompt_v1.txt', 'r', encoding='utf8') as file:
                    prompt_text = file.read()
                prompt_text = prompt_text.format(sep_ques=sep_ques)
                prompt = [prompt_text,]
                for uploaded_file in uploaded_files:
                    prompt.append(uploaded_file)
                response = model.generate_content(
                    prompt, generation_config=genai.types.GenerationConfig(temperature=0.0,))
                logging.info("Response successfully generated")
                try:
                    final_response = json.loads(response.text)
                    return final_response
                except Exception as e:
                    logging.error(
                        f"Format of gemini response is not as expected: {e}")
                    print(response.text)
            else:
                model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest",
                                            generation_config={"response_mime_type": "application/json"})
                with open('prompts/ontology_prompt_v2.txt', 'r', encoding='utf8') as file:
                    prompt_text = file.read()
                prompt = [prompt_text,]
                for uploaded_file in uploaded_files:
                    prompt.append(uploaded_file)
                response = model.generate_content(
                    prompt, generation_config=genai.types.GenerationConfig(temperature=0.0,))
                logging.info("Response successfully generated")
                try:
                    final_response = json.loads(response.text)
                    return final_response
                except Exception as e:
                    logging.error(
                        f"Format of gemini response is not as expected: {e}")
                    print(response.text)

        except Exception as e:
            logging.error(f"Error in gemini call: {e}")


class OntologyBuilder:
    @staticmethod
    def build_ontology(final_answer, all_entities):
        for mp in all_entities:
            if not isinstance(mp, dict):
                logging.warning(f"Unexpected type: {type(mp)}")
                continue

            for k, v in mp.items():
                if not isinstance(v, dict):
                    logging.warning(f"Unexpected type: {type(v)}")
                    continue
                if 'example' not in v or not isinstance(v['example'], str):
                    logging.warning(f"Unexpected format in: {v}")
                    continue

                if k in final_answer:
                    final_answer[k]['example'] += ', ' + v['example']
                else:
                    final_answer[k] = v

        return final_answer

    @staticmethod
    def save_ontology(final_answer, directory='output', filename_prefix='ontology'):

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logging.info(f"Directory '{directory}' created.")
            except Exception as e:
                logging.error(f"Error creating directory '{directory}': {e}")
                raise e

        timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{filename_prefix}_{timestamp}.json"
        file_path = os.path.join(directory, filename)

        try:
            with open(file_path, 'w', encoding='utf8') as json_file:
                json.dump(final_answer, json_file, indent=4)
            logging.info(f"Ontology saved to {file_path}")
        except Exception as e:
            logging.error(f"Error saving ontology: {e}")
            raise e

        return file_path
