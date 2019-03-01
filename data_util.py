# -*- coding: utf-8 -*-

import json

path = './data/baike/baike_qa_valid.json'
with open(path, 'r', encoding='utf-8') as file:
    for line in file.readlines():
        print(type(json.loads(line)))
