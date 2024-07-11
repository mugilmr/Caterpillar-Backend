import ollama
from flask import Flask,jsonify
from flask_cors import CORS, cross_origin
from predict import predict_failure
import numpy as np
from datetime import datetime
import pandas as pd
app = Flask(__name__)
CORS(app)

@app.route('/',methods=['POST','GET'])
def home():
    result = predict_failure('copy_Problem_Statement_2_ Data_set.xlsx')
    result = result.where(pd.notnull(result), None)
    prediction=[]
    for item in result.to_dict(orient='records')[0:10]:
        if item['target']=='Safe' and item['Probability of Failure']=='High':
            continue
        else:
            if item['target'] == 'Safe':
                prompt = f"Give me recommendations to maintain the {item["param_mapping_x"]} in my {item["Machine"]}."
            else:
                if np.isnan(item["high thres"]):
                    prompt = f"Let's say {item["param_mapping_x"]} of a {item["Machine"]} is {item["Value_x"]} and it has a probability of failure if it goes below {item["low thres"]}. Give recommendations to improve it."
                elif np.isnan(item["low thres"]):
                    prompt = f"Let's say {item["param_mapping_x"]} of a {item["Machine"]} is {item["Value_x"]} and it has a probability of failure if it exceeds below {item["high thres"]}. Give recommendations to improve it."
                else:
                    prompt = f"Let's say {item["param_mapping_x"]} of a {item["Machine"]} is {item["Value_x"]} and it has a probability of failure if it goes below {item["low thres"]} and exceeds {item["high thres"]}. Give recommendations to improve it."
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            del item['low thres']
            del item['high thres']
            item['suggestion'] = response['message']['content']
            item['Time'] = item['Time'].strftime("%Y-%m-%d %H:%M:%S")
            print(response)
            prediction.append(item)

    return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=True)


#
# begin=time.time()
# response = ollama.chat(model='llama3', messages=[
#   {
#     'role': 'user',
#     'content': "Let's say engine temperature of a machine is 104 and it has a high probability of failure if it exceeds 105. Give recommendations to improve it.",
#   },
# ])
#
# end=time.time()
#
# print(response['message']['content'])
#
# print(end-begin)

