'''
文件checking_outputs.json是由这个脚本生成的
'''
from litellm import embedding
import os
from pathlib import Path
from refchecker import LLMExtractor, LLMChecker
from ragchecker import RAGResults, RAGChecker
from ragchecker.metrics import all_metrics

class CheckerEnv:
    def __init__(self) -> None:
        self.model = None

    def xinferencesetup(self, modelname=None) -> None:
        # env variable
        os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"
        os.environ['XINFERENCE_API_KEY'] = "sk-72tkvudyGLPMi"  # [optional] no api key required
        self.model = 'xinference/qwen2.5-instruct'
        if modelname:
           self.model = modelname

if __name__ == "__main__":
    #LLM Env set
    envSetup = CheckerEnv()
    envSetup.xinferencesetup()

    # initialize ragresults from json/dict
    with open("checking_inputs.json") as fp:
    #with open("test.json") as fp:
        rag_results = RAGResults.from_json(fp.read())

        # set-up the evaluator
        evaluator = RAGChecker(
            extractor_name=envSetup.model,
            checker_name=envSetup.model,
            batch_size_extractor=32,
            batch_size_checker=32,
            extractor_api_base=os.environ['XINFERENCE_API_BASE'],
            checker_api_base=os.environ['XINFERENCE_API_BASE'],
            openai_api_key=os.environ['XINFERENCE_API_KEY']
        )

        # evaluate results with selected metrics or certain groups, e.g., retriever_metrics, generator_metrics, all_metrics
        evaluator.evaluate(rag_results, all_metrics, "checking_outputs.json")
        print(rag_results)