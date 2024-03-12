from transformers import AutoTokenizer
import requests
import json
import ast
import re


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# This function enables us to send a prompt to our LLM API.
def request_api(prompt,temperature):
    API_ENDPOINT = "" 
    data = {
            "prompt": prompt,
            "temperature":temperature,
            "stream": False,
            "max_tokens": 8040,
            "repeat_penalty": 1.2
        }
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data), stream=False)
    response_json_3 = response.json()

    return response_json_3['choices'][0].get('text', '')


# This function sends a part of a text file as a prompt to the LLM and receives a list of questions about the text.
def generate_questions(text,temperature):
    prompt_2 = f"""
    Your assignment involves a comprehensive examination of the given text, paying close attention to its significant aspects and particulars. 
    Upon completing your analysis, you are to compile a list of 5 questions that pertain closely to the text. 
    These questions should not only reference the title of the text but also be designed so that their answers are deducible from the text you've scrutinized. 
    
    ####
        Guidelines for Assessment:

        - Analyze the text carrefully.
        - You are to provide 5 questions about the Defense Appropriations of Act 2024, you must provide and mention the Defense Appropriations of Act 2024  or H.R. 4365  in your questions.
        - Format the response as a list of strings, with each string representing a question. For example: ['What is the total budget allocated in the Defense Appropriations Act of 2024?', ...]
    
        ####
        Here are some examples:
        
        
        Text for Analysis:
        Title: World Cup 2022
        Content: On May 12, 2022, France secured a 3-0 victory against Argentina in the World Cup 2022, with Mbappe scoring a hat-trick in the final 10 minutes of the game. He then exchanged a handshake with Messi before leaving the stadium.
        Questions:
        ["What date was the World Cup 2022 match between France and Argentina?", "What was the final score of the France vs. Argentina match in the World Cup 2022?", "How many goals did Mbappe score in the World Cup 2022 match against Argentina?","After the WC game between France and Argentina, what did mbappe did with Messi?"]
        
        Text for Analysis:
        Title: Department of Farmers Budget Act 2024, H.R. 9865
        Content: On December 16, 2024, the United States Senate received and discussed the act, subsequently placing it on the legislative calendar. The act stipulates that farmers are eligible for an annual assistance of $23,000 if they meet certain criteria: avoiding environmentally harmful products, using only American-made products, having at least three children, and earning less than $200,000 a year.
        Sample Questions:

        ["What is H.R. 9865 known as?", "When was the  H.R. 9865 discuted?","What are the conditions under the Department of Farmers Budget Act 2024 for farmers to receive assistance?", "How much financial assistance can a farmer receive annually under the Department of Farmers Budget Act 2024?","Accoridng to the Department of Farmers Budget Act 2024, can a farmer get assistance if he earns more than 100K a year ?"]
        ###
            
        <<<
        
        Text for Analysis:
        Title: Department of Defense Appropriations Act of 2024, H.R. 4365 
        Content:
        {text}

        Sample Questions:
        >>>
        """
    messages = [{"role": "user", "content": prompt_2}]
    prompt_api_2 = tokenizer.apply_chat_template(messages, tokenize=False)
    sentences =request_api(prompt_api_2,temperature)
    

    # Your input string
    input_str = sentences

    # Converting the string to a list
    try: 
        result_list = ast.literal_eval(input_str)
        return result_list
    except:
        corrected_list = agent_correction(input_str)
        result_list = ast.literal_eval(corrected_list)
        print('La liste corrigé a été réussi')
        return result_list
    
    

# Sometimes, the LLM does not return a list as an answer. This function will create an agent that handle such cases and ensure we get a list of questions.
def agent_correction(error_content):
    prompt_2=f"""
            [INST]

            You will receive a string  that may contain numerical prefixes, bad characters, or formatting issues, extract a clean list of questions. 
            The output should only include the questions, formatted as a list of strings, and should exclude any preceding numbers, non-list formats, or unrelated text.



            #### 
            Here's some exemple:


            Input: '1. "What are the main goals of the new policy?" 2. "How will the policy impact the budget for 2024?"'
            Output:["What are the main goals of the new policy?", "How will the policy impact the budget for 2024?"]

            Input: '2. s["Who is robert Kennedy?", "Capital of Germany?"] ln.'
            Output:["Who is robert Kennedy??", "Capital of Germany?']

            Input: 'Question 1: "What measures are included in the act?" Some irrelevant text. Question 2: "What is the deadline for implementation?"'
            Output:["What measures are included in the act?", "What is the deadline for implementation?"]

            ####

            Here's the ouput you receive: 
            Input: {error_content}
            Output:
            [/INS]
"""
    messages = [{"role": "user", "content": prompt_2}]
    prompt_api_2 = tokenizer.apply_chat_template(messages, tokenize=False)
    sentences =request_api(prompt_api_2,temperature=0)
    return sentences    

def generate_answer(texte, question, temperature):
    prompt = f"""
    Analyse the following text which is a part of the Department of Defense Appropriations Act of 2024, H.R. 4365 
    {texte}

    Then answer to the following question:
    {question}
    """
    messages = [{"role": "user", "content": prompt}]
    prompt_api = tokenizer.apply_chat_template(messages, tokenize=False)
    answer =request_api(prompt_api,temperature)
    return answer


# Open the file using 'with' to ensure it gets closed properly after reading
with open(r'/home/ghaith_ia23/Fine_tunning_LLM/us_law.txt', 'r') as file:
    content = file.read()
 
# Split the content based on the defined pattern for titles
sections = re.split(r'(?=TITLE\s+[IVXLCDM]+\s)', content)

divider = [5, 4, 3, 2, 1]
temperature_llm = [0,0.3]
input = []
output = []
for number_section in range(len(sections)):
    print('nous sommes à la sections: ', number_section)
    tokens = sections[number_section].split()
    if number_section == 0:
        print('OOOPS')
        divider = [1]
    else:
        divider = [4, 2, 1]
          
    start_index = 0  
    
    for div in divider:  
        print(f"Chunk with {div} parts:") 
        number_token = len(tokens) // div
        start_index = 0  
        end_index = 0 
        for i in range(div):  
            print(f"Chunk with {div} parts:", " we are in part number: ", i)
            end_index = start_index + number_token 
            current_tokens = tokens[start_index:end_index]
            text_chunk = ' '.join(current_tokens) 
            start_index = end_index
            for temperature in temperature_llm:
                if temperature == 0.3:
                    print('temperature 0.3')
                    for i in range(4):
                        print('niveau:', i)
                        try: 
                            questions = generate_questions(text_chunk,temperature)
                            for ques in questions:
                                
                                answer = generate_answer(text_chunk,ques, temperature)
                                
                                input.append(ques)
                                
                                output.append(answer)
                        except Exception as e:
                            print(f'An error occurred: {e}')
                            continue
                else:
                    print('Temperature 0:')

                    try:
                        questions = generate_questions(text_chunk,temperature)
                        for ques in questions:
                            
                            answer = generate_answer(text_chunk,ques, temperature)
                            input.append(ques)
                            output.append(answer)
                    except Exception as e:
                            print(f'An error occurred: {e}')
                            continue
            
# This script creates a .jsonl file containing our input and output data to form the final dataset.             
json_lines = []
for inp, out in zip(input, output):
    json_line = {"input": inp, "output": out}
    json_lines.append(json.dumps(json_line))

with open('data.jsonl', 'w') as file:
    for line in json_lines:
        file.write(line + '\n')
                        
        
