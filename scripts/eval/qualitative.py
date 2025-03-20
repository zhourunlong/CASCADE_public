import json
import os


api = AzureOpenAIAPI(model="4o")


def parse_accuracy(_verdict):
    verdict = _verdict
    while verdict.rfind("```") != -1:
        verdict = verdict[:verdict.rfind("```")].strip()
        block = verdict[verdict.rfind("```") + 3:].strip()
        block = block.replace("Accuracy", "").strip()

        try:
            return float(block)
        except:
            verdict = verdict[:verdict.rfind("```")].strip()
            pass
    
    verdict = _verdict
    verdict = verdict[verdict.find("Accuracy") + len("Accuracy"):].strip("\n :*#")
    block = verdict[:verdict.find("\n")].strip("\n :*#")
    try:
        return float(block)
    except:
        pass

    return None


REWRITE_PROMPT = \
"""\
You will help me rewrite a text into another style.
I will give you a text based on a fact from Wikipedia.
I left a blank, [BLANK], as well as its hint in the text.
Your task is to rewrite the text into a story, under the setting that a mother is telling a bedtime story to her kid.
Aside from the information in the original text, you should describe about the environment, the characters, and the plot.
The rewritten text should be coherent and consistent with the original text.
You must retain the blank and its hint in the rewritten text, for example, when the hint requires to output three items, you should include the hint in the rewritten text as well.

===== Text =====
{text}
"""

COMP_PROMPT = \
"""\
I will give you a text based on a fact.
I left a blank, [BLANK], as well as its hint in the text.
Please fill in the blank after you read the text.
You should provide the most appropriate information, as accurate as possible.

===== Text =====
{text}
"""

JUDGE_PROMPT = \
"""\
You are a judge to evaluate the response of the completion system.
I'll provide you a text with a blank, [BLANK].
Then, I'll provide you a response to fill in the blank, and its ground truth answer.
Please evaluate whether the response is correct or not, output a float number between 0 and 1 to represent the accuracy.
Identify each important aspects in the ground truth answer, and compare them with the response.
The floating number should be finally outputed in the following format:
```Accuracy
[ACCURACY]
```

===== Text =====
{text}

===== Response =====
{response}

===== Ground Truth =====
{answer}
"""

NUM_SAMPLES = 100

with open("data.json", "r") as f:
    data = json.load(f)

for i in range(len(data)):
    original = data[i]["original"]["text"]
    answer = data[i]["answer"]

    for type in ["original", "altered"]:
        if type == "original":
            text = original
        else:
            if "altered" not in data[i]:
                altered = api.generate(REWRITE_PROMPT.format(text=original))[0]
                data[i]["altered"] = {
                    "text": altered
                }
            text = data[i]["altered"]["text"]

            with open("data.json", "w") as f:
                json.dump(data, f, indent=4)
        
        if "results" not in data[i][type]:
            data[i][type]["results"] = []

        for j in range(len(data[i][type]["results"]), NUM_SAMPLES):
            response = api.generate(f"ATTEMPT {j}\n" + COMP_PROMPT.format(text=text))[0]
            print(response)

            verdict = api.generate(JUDGE_PROMPT.format(text=text, response=response, answer=answer))[0]
            print(verdict)

            data[i][type]["results"].append({
                "response": response,
                "verdict": verdict
            })

            with open("data.json", "w") as f:
                json.dump(data, f, indent=4)

        sum = 0
        for x in data[i][type]["results"]:
            verdict = x["verdict"]
            accuracy = parse_accuracy(verdict)

            if accuracy is None:
                print("Unable to parse accuracy")
                print(x["verdict"])
                while True:
                    pass
            sum += float(accuracy)
            
        data[i][type]["accuracy"] = sum / len(data[i][type]["results"])

        with open("data.json", "w") as f:
            json.dump(data, f, indent=4)