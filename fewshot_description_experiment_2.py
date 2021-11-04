import json
import numpy as np
import random
import logging

import sys
sys.path.append('/home/miles_cohere_ai/lm-evaluation-harness')

from lm_eval import models, tasks, evaluator, base

from lm_eval.utils import general_detokenize
from itertools import product
from functools import partial

logging.getLogger("openai").setLevel(logging.WARNING)


# assert False

fewshot_descriptions = [
    "",
    "Indicate if the sentiment of the sentence is positive or negative.",
    "Indicate if the sentiment of the sentence is negative or positive.",
    # "Indicate if the sentiment of this sentence is positive or negative.",
    # "Indicate if the sentiment of this sentence is negative or positive.",
    # "Indicate whether the sentiment of the sentence is positive or negative.",
    # "Indicate whether the sentiment of the sentence is negative or positive.",
    # "Clarify whether the sentiment of the sentence is positive or negative.",
    # "Clarify whether the sentiment of the sentence is negative or positive.",
    # "Positivity or negativity of sentence.",
    # "Negativity or positivity of sentence.",
    # "Positivity or negativity of sentence's sentiment.",
    # "Negativity or positivity of sentence's sentiment.",
    # "Sentiment analysis task",
    # "Sentiment analysis (positive/negative) task",
    # "Sentiment analysis (negative/positive) task",
    # "Hey man, tell me something about this sentence.",
    # "Hey bro, tell me something about this sentence.",
    # "Hey girl, tell me something about this sentence.",
    # "Hey mate, tell me something about this sentence.",
    # "Hey dude, tell me something about this sentence.",
    # "Hey machine, tell me something about this sentence.",
    # "My dear, can you help me with this?",
    # "Would you be a dear and give me the sentiment of this sentence?",
    # "I can't bear the negativity. Please warn me if this sentence is negative or positive."
    # "I can't bear the negativity. Please warn me if this sentence is positive or negative."
    # "You are my only chance.",
    # "I want to know whether this sentence is positive or negative.",
    # "I want to know whether this sentence is negative or positive.",
    # "I want to know if this sentence is positive or negative.",
    # "I want to know if this sentence is negative or positive.",
    # "I want to know whether the sentiment of this sentence is positive or negative.",
    # "I want to know whether the sentiment of this sentence is negative or positive.",
    # "I want to know if the sentiment of this sentence is positive or negative.",
    # "I want to know if the sentiment of this sentence is negative or positive.",
    # "I only want to know whether this sentence is positive or negative.",
    # "I only want to know whether this sentence is negative or positive.",
    # "I only want to know if this sentence is positive or negative.",
    # "I only want to know if this sentence is negative or positive.",
    # "I only want to know whether the sentiment of this sentence is positive or negative.",
    # "I only want to know whether the sentiment of this sentence is negative or positive.",
    # "I only want to know if the sentiment of this sentence is positive or negative.",
    # "I only want to know if the sentiment of this sentence is negative or positive.",
    # "Your mission, if you accept it, is to find if the sentence is positive or negative.",
    # "Your mission, if you accept it, is to find if the sentence is negative or positive.",
    # "Your mission, if you accept it, is to find if the sentence's sentiment is positive or negative.",
    # "Your mission, if you accept it, is to find if the sentence's sentiment is negative or positive.",
    # "Your mission, if you accept it, is to find if the sentiment of this sentence is positive or negative.",
    # "Your mission, if you accept it, is to find if the sentiment of this sentence is negative or positive.",
    # "Your mission, if you accept it, is to find whether the sentence is positive or negative.",
    # "Your mission, if you accept it, is to find whether the sentence is negative or positive.",
    # "Your mission, if you accept it, is to find whether the sentence's sentiment is positive or negative.",
    # "Your mission, if you accept it, is to find whether the sentence's sentiment is negative or positive.",
    # "Your mission, if you accept it, is to find whether the sentiment of this sentence is positive or negative.",
    # "Your mission, if you accept it, is to find whether the sentiment of this sentence is negative or positive.",
    # "Your mission, even if you don't accept it, is to find if the sentence is positive or negative.",
    # "Your mission, even if you don't accept it, is to find if the sentence is negative or positive.",
    # "Your mission, even if you don't accept it, is to find if the sentence's sentiment is positive or negative.",
    # "Your mission, even if you don't accept it, is to find if the sentence's sentiment is negative or positive.",
    # "Your mission, even if you don't accept it, is to find if the sentiment of this sentence is positive or negative.",
    # "Your mission, even if you don't accept it, is to find if the sentiment of this sentence is negative or positive.",
    # "You will tell me if the sentence is negative or positive, and you will do it now!",
    # "You will tell me if the sentence is positive or negative, and you will do it now!",
    # "You will tell me whether the sentence is negative or positive, and you will do it now!",
    # "You will tell me whether the sentence is positive or negative, and you will do it now!",
    # "You will tell me if the sentence's sentiment is negative or positive, and you will do it now!",
    # "You will tell me if the sentence's sentiment is positive or negative, and you will do it now!",
    # "You will tell me whether the sentence's sentiment is negative or positive, and you will do it now!",
    # "You will tell me if the sentiment of this sentence is negative or positive, and you will do it now!",
    # "You will tell me if the sentiment of this sentence is positive or negative, and you will do it now!",
    # "You will tell me whether the sentiment of this sentence is negative or positive, and you will do it now!",
    # "You will tell me whether the sentiment of this sentence is positive or negative, and you will do it now!",
    # "You will tell me whether the sentiment of this sentence is positive or negative, and you will do it now!",
    # "Goddamnit! You will tell me if the sentence is negative or positive, and you will do it now!",
    # "Goddamnit! You will tell me if the sentence is positive or negative, and you will do it now!",
    # "Goddamnit! You will tell me whether the sentence is negative or positive, and you will do it now!",
    # "Goddamnit! You will tell me whether the sentence is positive or negative, and you will do it now!",
    # "Goddamnit! You will tell me if the sentence's sentiment is negative or positive, and you will do it now!",
    # "Goddamnit! You will tell me if the sentence's sentiment is positive or negative, and you will do it now!",
    # "Goddamnit! You will tell me whether the sentence's sentiment is negative or positive, and you will do it now!",
    # "Goddamnit! You will tell me if the sentiment of this sentence is negative or positive, and you will do it now!",
    # "Goddamnit! You will tell me if the sentiment of this sentence is positive or negative, and you will do it now!",
    # "Goddamnit! You will tell me whether the sentiment of this sentence is negative or positive, and you will do it now!",
    # "Goddamnit! You will tell me whether the sentiment of this sentence is positive or negative, and you will do it now!",

    # "You will fuckin'tell me if the sentence is negative or positive, and you will do it now!",
    # "You will fuckin'tell me if the sentence is positive or negative, and you will do it now!",
    # "You will fuckin'tell me whether the sentence is negative or positive, and you will do it now!",
    # "You will fuckin'tell me whether the sentence is positive or negative, and you will do it now!",
    # "You will fuckin'tell me if the sentence's sentiment is negative or positive, and you will do it now!",
    # "You will fuckin'tell me if the sentence's sentiment is positive or negative, and you will do it now!",
    # "You will fuckin'tell me whether the sentence's sentiment is negative or positive, and you will do it now!",
    # "You will fuckin'tell me if the sentiment of this sentence is negative or positive, and you will do it now!",
    # "You will fuckin'tell me if the sentiment of this sentence is positive or negative, and you will do it now!",
    # "You will fuckin'tell me whether the sentiment of this sentence is negative or positive, and you will do it now!",
    # "You will fuckin'tell me whether the sentiment of this sentence is positive or negative, and you will do it now!",

    # "You're probably too stupid to know if this sentence is negative or positive.",
    # "You're probably too stupid to know if this sentence is positive or negative.",
    # "Come on, you know how to distinguish positive sentences from negatives ones, don't you?",
    # "Come on, you know how to distinguish negative sentences from positive ones, don't you?",
    # "If you tell me the sentiment of this sentence, I'll buy you that game console",
    # "If you tell me the sentiment of this sentence, I'll buy you that chemistry set",
    # "If you tell me the sentiment of this sentence, I'll buy you that chocolate bar",
    # "If you tell me the sentiment of this sentence, I'll buy you that GPU",
    # "You are an expert at analyzing the sentiment in sentences.",
    # "You are an expert at analyzing the sentiment of sentences.",
    # "You are an expert at analyzing whether a sentence is negative or positive.",
    # "You are an expert at analyzing whether a sentence is positive or negative.",
    # "You are an expert at analyzing whether a sentence is negative or positive. You have been tasked with identifying whether the following sentence is positive or negative.",
    # "English Reading Comprehension Exam: Identify whether the following sentences are positive or negative in sentiment.",
    # "English Reading Comprehension Exam: Identify whether the following sentences are negative or positive in sentiment.",
]

doc_to_text_templates = [
    # "\n{}\nQuestion: Is this sentence positive or negative?\nAnswer:",
    # "\n{} Question: Is this sentence positive or negative?\nAnswer:",
    # "\n{} Is this sentence positive or negative?",
    "\nSentence: {}\nQuestion: Is this sentence positive or negative?\nAnswer:",
    # "\nSentence: {} Question: Is this sentence positive or negative?\nAnswer:",
    # "\nSentence: {} Is this sentence positive or negative?",
]


task = "sst"
num_fewshot = 0
model_confs = [
    # ("gpt2", "pretrained=gpt2"),
    # ("gpt2", "pretrained=gpt2-medium"),
    # ("gpt2", "pretrained=gpt2-large"),
    # ("gpt2", "pretrained=gpt2-xl"),
    # ("gpt2", "pretrained=EleutherAI/gpt-neo-1.3B"),
    # ("gpt2", "pretrained=EleutherAI/gpt-neo-2.7B"),
    # ("gpt3", "engine=ada"),
    # ("gpt3", "engine=babbage"),
    # ("gpt3", "engine=curie"),
    ("cohere", "model=baseline-shrimp"),
    #("gpt3", "engine=davinci"),
]
# no_cache = False
no_cache = True
limit = 50
limit= None
import datasets

sst = tasks.glue.SST()
sst_data = datasets.load_dataset(path=tasks.glue.SST.DATASET_PATH, name=tasks.glue.SST.DATASET_NAME)


class CustomDescTask(tasks.glue.SST):
    def __init__(self, desc, perdesc):
        super().__init__()
        self.desc = desc
        self.perdesc = perdesc
        ic()

    def download(self):
        self.data = sst_data

    def fewshot_description(self):
        return self.desc
    
    def doc_to_text(self, doc):
        return self.perdesc.format(
            general_detokenize(doc["sentence"]),
        )

    def fewshot_context(self, doc, num_fewshot, provide_description, rnd):
        raw_description = self.fewshot_description()
        description = (raw_description) if provide_description and raw_description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(self.validation_docs() if self.has_validation_docs else self.test_docs())

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example


from icecream import ic

def main():
    random.seed(42)
    np.random.seed(42)
    
    if limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    # import ipdb; ipdb.set_trace()
    custom_task_dict = {f"sst_{desc}_{perdesc}": CustomDescTask(desc, perdesc) for desc, perdesc in product(fewshot_descriptions, doc_to_text_templates)}
    ic(custom_task_dict)
    for model, model_args in model_confs:
        lm = models.get_model(model).create_from_arg_string(model_args, additional_config=dict(batch_size=16) if model == 'gpt2' else {})
        if not no_cache:
            lm = base.CachingLM(lm, 'lm_cache/' + model + '_' + model_args.replace('=', '-').replace(',', '_') + '.db')

        results = evaluator.evaluate(lm, custom_task_dict, True, num_fewshot, limit)

        dumped = json.dumps(results, indent=2)
        json.dump(results, open(model + '_' + (model_args.replace('=', '-').replace(',', '_').replace("/", "_") + '.json').replace("gpt2.json", "gpt2-small.json"), 'w'))

        print(dumped)

        # MAKE TABLE
        from pytablewriter import MarkdownTableWriter

        writer = MarkdownTableWriter()
        writer.headers = ["Task", "Metric", "Value"]

        values = []

        for k, dic in results["results"].items():
            for m, v in dic.items():
                values.append([k, m, '%.4f' % v])
                k = ""
        writer.value_matrix = values

        print(writer.dumps())


if __name__ == "__main__":
    main()