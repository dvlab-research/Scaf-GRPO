# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import openai
from openai import OpenAI, OpenAIError

os.environ["OPENAI_BASE_URL"] = "https://api.ai-gaochao.cn/v1"
os.environ["OPENAI_API_KEY"] = "sk-gsGtodWPMMzq3TYA13289184Dd4c4e35832201F375130dA5"  # sitong


try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

from symeval import EvaluatorMathBatch
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


evaluator = EvaluatorMathBatch()




def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    
    if ret_score == 0.0:
        pred_answer = None
        try:
            string_in_last_boxed = last_boxed_only_string(model_output)
            if string_in_last_boxed is not None:
                pred_answer = remove_boxed(string_in_last_boxed)

            if pred_answer is not None:
                # symeval
                try:
                    ret_score = eval_math_symeval(ground_truth, pred_answer)
                    ret_score = float(ret_score)
                except Exception:
                    pass
                
                # api
                if ret_score == 0.0:
                    try:
                        ret_score = eval_math_api(ground_truth, pred_answer, api_model_name='gpt-4o-mini')
                    except openai.APITimeoutError as e:
                        print(f"OpenAI API TimeoutError: {e}")
                    except OpenAIError as e:
                        print(f"OpenAI Error: {e}")
        except:
            pass
               
    return ret_score



def eval_math_symeval(gt_answer, pred_answer):
    score = evaluator.batch_eq(ref_answers=[gt_answer], pred_answers=[pred_answer])[0]
    score = float(score)
    return score


# ---------------------------------------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

def chat_api(model, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    if model in ['deepseek-r1', 'deepseek-reasoner', 'deepseek-r1-250120']:
        thinking = response.choices[0].message.reasoning_content
        answer = response.choices[0].message.content
        output = (thinking, answer)
    else:
        output = response.choices[0].message.content
    return output
    
def parse_answer_boxed(pred_str):
    ## check fail case-1
    if 'boxed' not in pred_str:
        return ""
    ## check fail case-2
    ans = pred_str.split("boxed")
    if len(ans) == 1:
        return ""
    ## check fail case-3
    ans = ans[-1]
    if len(ans) == 0:
        return ""
    ##
    try:
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
    except:
        return ""
    a = a.strip(' \n').strip(' \n').strip(' \n')
    return a


def eval_math_api(gt_answer, pred_answer, api_model_name):
    # get message
    prefix = "Given a 'Student Answer' and a 'Ground-Truth Answer, your task is to verify the correctness of the 'Student Answer' using 'Ground-Truth Answer' as a reference. If the 'Student Answer' is same as the 'Ground-Truth Answer', then it can be regarded as correct.\nPlease put your final decision (correct or wrong) in \\boxed{}, such as \\boxed{correct} or \\boxed{wrong}."
    user_content = prefix + '\n' + 'Student Answer: ' + pred_answer + '\n' + 'Ground-Truth Answer: ' + str(gt_answer)
    message = [
        {"role": "user", "content": user_content},
    ]
    # 
    try:
        output = chat_api(api_model_name, message)
    except:
        output = ""
    
    # parse pred from \\boxed{}
    pred = parse_answer_boxed(output)
    pred = pred.lower()
    
    # eval
    if 'correct' in pred:
        score = 1.0
    else:
        score = 0.0        

    return score

