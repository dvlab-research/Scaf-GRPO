from symeval import EvaluatorMathBatch
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

evaluator = EvaluatorMathBatch()



def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if eval_math_symeval(ground_truth, answer):
                retval = 1.0
    except Exception as e:
        print(e)
        
    return retval



def eval_math_symeval(gt_answer, pred_answer):
    score = evaluator.batch_eq(ref_answers=[gt_answer], pred_answers=[pred_answer])[0]
    return score
