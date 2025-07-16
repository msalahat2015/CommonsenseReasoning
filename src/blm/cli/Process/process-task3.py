
import os
import logging
import argparse
from datasets import  load_dataset,Dataset, DatasetDict, Features, Value, Sequence, ClassLabel
from blm.utils.helpers import logging_config
import pyarrow as paPyright



logger = logging.getLogger(__name__)
def some_function():
    print("Hello")

def save_dataset(dataset, output_path, n):
    """
    Convert the dataset into messages as follows, each training example will
    follow the following format:
      {
        "messages": [{"content": "<SYSTEMPROMPT>", "role": "system"},
                     {"content": "<INSTRUCTIONS>", "role": "user"},
                     {"content": "<RESPONSE>", "role": "assistant"}
                    ]
      }
    :param dataset: Dataset
    :param output_path: str - output path to save dataset to
    :param n: int - number of training examples to sample
    :return: Dataset
    """

    # Open the system prompt file and read its content
    with open('/rep/msalahat/LLMTraining/mysrc/blm/prompts/Task3/system_prompt.txt', 'r') as file:
        system_prompt = file.read()

    # Open the system prompt file and read its content
    with open('/rep/msalahat/LLMTraining/mysrc/blm/prompts/Task3/user_prompt.txt', 'r') as file:
        user_prompt = file.read()
    
   

    train_messages=[
        {  # This is a dictionary
            "messages": [
                {"content": system_prompt, "role": "system"},
                {
                    "content": user_prompt.format(
                        statement=e['non-commonsense-statement']
                       
                    ),
                    "role": "user"
                },
                {"content": f"{e['reason1']}\n"
                            f"{e['reason2']}\n"
                            f"{e['reason3']}\n" , "role": "assistant"}
            ]
        }
    for e in dataset["train"]
   ]
    print(train_messages[0])

    eval_messages=[
        {  # This is a dictionary
            "messages": [
                {"content": system_prompt, "role": "system"},
                {
                    "content": user_prompt.format(
                        statement=e['non-commonsense-statement']
                       
                    ),
                    "role": "user"
                },
                {"content": f"{e['reason1']}\n"
                            f"{e['reason2']}\n"
                            f"{e['reason3']}\n" , "role": "assistant"}
            ]
        }
    for e in dataset["validation"]
   ]

    test_messages=[
        {  # This is a dictionary
            "messages": [
                {"content": system_prompt, "role": "system"},
                {
                    "content": user_prompt.format(
                        statement=e['non-commonsense-statement']
                    ),
                    "role": "user"
                },
                {"content": "", "role": "assistant"}
            ],
            "label":f"{e['reason1']}\n"
                    f"{e['reason2']}\n"
                    f"{e['reason3']}\n"
        }
    for e in dataset["test"]
   ]
   
   
    
   
    #train_messages = [{"messages": [{"content": "", "role": "system"},{"content": e["sentence1"], "role": "user"}, {"content": e["label"], "role": "assistant"}]} for e in dataset["train"]]
    #eval_messages = [{"messages": [{"content": e["sentence1"], "role": "user"}, {"content": e["label"], "role": "assistant"}]} for e in dataset["test"]]
   # Define the features of the dataset explicitly, ensuring correct data types
        # Define the features using Features and Value/Sequence from datasets library
    # features = Features({
    #     "messages": Sequence({
    #         "content": Value("string"),
    #         "role": Value("string")
    #     })
    # })

    # ds = DatasetDict({
    #     "train": Dataset.from_list(train_messages, features=features),
    #     "eval": Dataset.from_list(eval_messages, features=features)
    # })

    ds = DatasetDict({
        "train": Dataset.from_list(train_messages),
        "eval": Dataset.from_list(eval_messages),
        "test": Dataset.from_list(test_messages)     
    })

    ds.save_to_disk(output_path)
    return ds


def main(args):
    #dataset = load_dataset("Kamyar-zeinalipour/ArabicSense") #'arbml/CIDAR"
    #dataset = dataset['train'].train_test_split(test_size=0.2)
    dataset = load_dataset(
    "csv", 
    data_files={
        "train": "/rep/msalahat/LLMTraining/data/task3_train.csv", 
        "test": "/rep/msalahat/LLMTraining/data/task3_test.csv", 
        "validation": "/rep/msalahat/LLMTraining/data/task3_validation.csv"
    }
    )
    
    ds = save_dataset(dataset, args.output_path, args.n)
    logger.info(f"Total training examples: {len(ds['train'])}")
    logger.info(f"Total eval examples: {len(ds['eval'])}")
    logger.info(f"Total test examples: {len(ds['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",type=str,default='output_folder' , help="Data output path")
    parser.add_argument("--n", type=int, default=500, help="Number of training examples to sample")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.output_path), exist_ok=True)
    os.makedirs(os.path.join("log"), exist_ok=True)
    logging_config("/rep/msalahat/LLMTraining/log/log.log")

    main(args)