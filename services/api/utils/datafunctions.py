from pathlib import Path
from sqlalchemy import create_engine, text
from api.utils.configs import *

engine = create_engine("postgresql:///:memory:", echo=True, future=True)

"""
DATABASE FUNCTIONS
"""
def get_description(label: int):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT description FROM styles WHERE label_id = :label"),
                {"label", label}
            )
            return result
    except Exception as e:
        print(e)

def read_class_txt(txt_file):
    # returns dictionary with key = id and value = label
    with open(txt_file) as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        #classes_dict = dict(lines)
        classes_list = [line[1] for line in lines]
        # classes_dict = {int(key):classes_dict[key] for key in classes_dict}
        return classes_list


