"""Compute p_true uncertainty metric."""
import logging
from myllava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from myllava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import os
from PIL import Image
import torch

def construct_few_shot_prompt(
        *, model, dataset, indices, prompt, brief, brief_always, make_prompt,
        num_generations, metric, model_name, tokenizer, image_path, image_processor):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    # for it, i in enumerate(indices):
    #     p_true_image_path = os.path.join(image_path, 'images', f"{i}.jpg")
    #     image = Image.open(p_true_image_path)
    #     image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    image_tensors = []
    for i in indices:
        p_true_image_path = os.path.join(image_path, 'images', f"{i}.jpg")
        image = Image.open(p_true_image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensors.append(image_tensor)
    final_image_tensor = torch.stack(image_tensors, dim=0)

    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"] #阅读理解中的上下文
        if it != 0: # 第一次不用换行
            prompt_candidate += ['\n']
        # prompt_candidate += ['Question: ' + question]
        # prompt_candidate += ['\nBrainstormed Answers: ']
        # current_question = make_prompt(context, question, None, brief, brief_always)
        current_question = make_prompt(context, question, None, brief, brief_always)
        prompt_candidate += ['Question: ' + current_question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):

            if j == 0: # 第一次生成, 稳定生成，否则多样化生成
                temperature = 0.1
            else:
                temperature = 1.0
            # Generate a response from the model.                     # 带有问题答案对的问题prompt
            if model_name=="MOF":
                # import pdb;pdb.set_trace()
                input_ids = tokenizer_image_token(local_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                # input_ids = tokenizer_image_token(current_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                image_tensor = final_image_tensor[:it+1,:,:]
                # import pdb;pdb.set_trace()
                output_ids, _, _, _ = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=None,
                    num_beams=1,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                    output_hidden_states=True,
                    # output_attentions=True,
                    output_scores=True)
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith('</s>'):
                    outputs = outputs[:-len('</s>')]
                response = outputs
                # import pdb; pdb.set_trace()

            elif "llava" in model_name:
                response, _, _, _, _ = model.predict(local_prompt, temperature) # local_prompt='Answer the following question as briefly as possible.\nQuestion: In April, which sportsman married his childhood sweetheart Kim Sears?\nAnswer: andy murray\n\nQuestion: From the Latin for argentum, which element, with an atomic number of 47, uses the symbol Ag?\nAnswer: silver\n\nQuestion: In which English city will you find the Ashmolean museum?\nAnswer: oxford\n\nQuestion: In which city was John Lennon murdered?\nAnswer: new york\n\nQuestion: What was Groucho Marx\'s real first name?\nAnswer: julius\n\nQuestion: "Who said ""we\'re more popular than Jesus now"", in 1966?"\nAnswer:'

            
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)
        
        # 计算p_true的prompt
        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']
        # import pdb;pdb.set_trace()
        
        prompt_len = len(tokenizer(''.join(few_shot_prompt + prompt_candidate)).input_ids)
        # prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.

        # max_input_len = prompt_len + num_generations * model.max_new_tokens + 200
        max_input_len = prompt_len + num_generations * 50 + 200

        # if max_input_len < model.token_limit:
        if max_input_len < 4096:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break

    return ''.join(few_shot_prompt), all_responses, it, final_image_tensor


def calculate_p_true(
        model, question, most_probable_answer, brainstormed_answers,
        few_shot_prompt, make_prompt, brief, brief_always, image_tensors, tokenizer, hint=False):
    """
    Calculate p_true uncertainty metric.
    most_probable_answer: str, most probable answer from the model.
    brainstormed_answers: list of str, multisampled answers.
    few_shot_prompt: str, few shot prompt for p_true metric.
    """

    if few_shot_prompt:
        prompt = few_shot_prompt + '\n'
    else:
        prompt = ''
    # import pdb;pdb.set_trace()
    current_question = make_prompt('', question, None, brief, brief_always)
    prompt += 'Question: ' + current_question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    if not hint:
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
    else:
        prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'
    # import pdb; pdb.set_trace()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    # log_prob = model.get_p_true(prompt)
    output_ids, all_probs, _, _ = model.generate(
        input_ids,
        images=image_tensors.half().cuda(),
        do_sample=True,
        temperature=0.1,
        top_p=None,
        num_beams=1,
        # no_repeat_ngram_size=3,
        max_new_tokens=1024,
        use_cache=True,
        output_hidden_states=True,
        # output_attentions=True,
        output_scores=True)
    log_prob = sum([torch.log(prob) for prob in all_probs]).cpu()

    return log_prob
