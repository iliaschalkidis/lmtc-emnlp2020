import json
import os
from document_model import Document


class JSONLoader:

    def __init__(self):
        super(JSONLoader).__init__()

    def read_file(self, filename: str):
        tags = []
        with open(filename) as file:
            data = json.load(file)
        sections = []
        text = ''
        sections.append(data['header'])
        sections.append(data['recitals'])
        sections.extend(data['main_body'])
        sections.extend(data['attachments'])

        text = '\n'.join(sections)
        for concept in data['concepts']:
            tags.append(concept)

        return Document(text, tags, filename=os.path.basename(filename))
