"""Data Loading Utilities."""
import os
import json
import hashlib
import datasets
import pandas as pd
from datasets import Dataset
from uncertainty.utils import utils
parser = utils.get_parser()
args, unknown = parser.parse_known_args()
# data :
# Dataset({
#     features: ['id', 'title', 'context', 'question', 'answers'],
#     num_rows: 130319
# })
def transform_answers(answer):
    return {'text': [answer], 'answer_start': [0]}

def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    user = os.environ['USER']

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }
        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])
        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
    elif dataset_name == "mmvp":
        benchmark_dir = os.path.join(args.image_path, 'Questions.csv')
        df = pd.read_csv(benchmark_dir)
        df['question'] = df['Question'] + " " + df['Options']
        df["answers"] = df["Correct Answer"]
        df = df.drop(['Question', 'Options', 'Correct Answer'], axis=1)
        df = df.rename(columns={'Index': 'id'})
        df['id'] = df['id'].astype(str)
        df['id'] = df['id'].apply(lambda x: x.replace('.0', ''))
        df['id'] = df['id'].apply(lambda x: x.replace(' ', ''))
        df['title'] = ''
        df['context'] = ''
        df['answers'] = df['answers'].apply(transform_answers)
        validation_dataset = Dataset.from_pandas(df)
        features = validation_dataset.features
        train_dataset = Dataset.from_pandas(df)
        # 创建空的 validation_dataset
        # empty_data = {feature: [] for feature in features}
        # train_dataset = Dataset.from_dict(empty_data, features=features)
        # import pdb;pdb.set_trace()

    else:
        raise ValueError

    return train_dataset, validation_dataset
