import os
import pdb
import openai
import requests
import json
import time
api_key = "sk-NwLOzmXkvJyeVeSp378840FdC1B840E4B4D7080a23A5B90e"
# used_url = "https://chatapi.onechats.top/v1/chat/completions"
used_url = "https://sapi.onechats.top/v1/chat/completions"
class OpenAIClient():
    def __init__(self,
                 model="gpt-4",
                 temperature=0.7,
                 system_prompt=""):

        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.url = used_url

    def chat(self, prompt):

        try:
            
            data = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature" : self.temperature
            })
            headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {api_key}",
            }
            
            response = requests.request("POST", self.url, headers=headers, data=data, timeout=(300, 300))
                # time.sleep(3)
            response_data = response.json()
            
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            print('TIMEOUT. Sleeping and trying again.')
            time.sleep(3)



def parse_object_goal_instruction(language_instr):
    """
    Parse language instruction into a series of landmarks
    Example: "first go to the kitchen and then go to the toilet" -> ["kitchen", "toilet"]
    """
    #TODO 不再调用openai
    # import openai

    # openai_key = os.environ["OPENAI_KEY"]
    # openai.api_key = openai_key
    question = f"""
    I: go to the kitchen and then go to the toilet. A: kitchen, toilet
    I: go to the chair and then go to another chair. A: chair, chair
    I: navigate to the green sofa and turn right and find several chairs, finally go to the painting. A: green sofa, chairs, painting
    I: approach the window in front, turn right and go to the television, and finally go by the oven in the kitchen. A: window, television, oven, kitchen
    I: walk to the plant first, turn around and come back to the table, go further into the bedroom, and stand next to the bed. A: plant, table, bedroom, bed
    I: go by the stairs, go to the room next to it, approach the book shelf and then go to the table in the next room. A: stairs, room, book shelf, table, next room
    I: Go front left and move to the table, then turn around and find a cushion, later stand next to a column before finally navigate to any appliances. A: table, cushion, column, appliances.
    I: Move to the west of the chair, with the sofa on your right, move to the table, then turn right 90 degree, then find a table. A: chair, table
    I: {language_instr}. A:"""
    #TODO 这里写自己的openai的调用方式
    openai_model=OpenAIClient(model="gpt-3.5-turbo",
                             temperature=0,
                             system_prompt="Please help complete the paragraph after A: as example")
    response=openai_model.chat(question)
    # response = openai.Completion.create(
    #     engine="text-davinci-002",
    #     prompt=question,
    #     max_tokens=64,
    #     temperature=0.0,
    #     stop=None,
    # )
    # result = response["choices"][0]["text"].strip()
    # pdb.set_trace()
    if "A:" in response:
        landmark_i = response.index("A:")
        response = response[landmark_i+2:].replace("A:", "").replace(" ","")
    if "." in response:
        response = response.replace(".","")
    print("landmarks: ", response)
    return [x.strip() for x in response.split(",")]


def parse_spatial_instruction(language_instr):
    # import openai
    openai_model=OpenAIClient(model="gpt-3.5-turbo",
                             temperature=0,
                             system_prompt="")
    # openai_key = os.environ["OPENAI_KEY"]
    # openai.api_key = openai_key
    # instructions_list = language_instr.split(",")
    instructions_list = [language_instr]
    results = ""
    for lang in instructions_list:
        question = f"""
# move a bit to the right of the refrigerator.
robot.move_to_right('refrigerator')
###
# move in between the couch and bookshelf.
robot.move_in_between('couch', 'bookshelf')
###
# move in the middle of the stairs and oven.
robot.move_in_between('stairs', 'oven')
###
# face the toilet.
robot.face('toilet')
###
# move to the south side of the table
robot.move_south('table')
###
# move to the west of the chair
robot.move_west('chair')
###
# move to the east of the chair
robot.move_east('chair')
###
# turn left 60 degrees
robot.turn(-60)
###
# turn right 20 degrees
robot.turn(20)
###
# find any chairs in the environment
robot.move_to_object('chair')
###
# come to the table
robot.move_to_object('table')
###
# with the television on your left
robot.with_object_on_left('television')
###
# with the toilet to your right, turn left 30 degrees
robot.with_object_on_right('toilet')
robot.turn(-30)
###
# with the toilet to your left, turn right 30 degrees
robot.with_object_on_left('toilet')
robot.turn(30)
###
# with the television behind you
robot.face('television')
robot.turn(180)
###
# go to the south of the table in front of you, with the table on your left, turn right 45 degrees
robot.move_south('table')
robot.with_object_on_left('table')
robot.turn(45)
###
# move forward for 3 meters
robot.move_forward(3)
###
# move right 2 meters
robot.turn(90)
robot.move_forward(2)
###
# move left 3 meters
robot.turn(-90)
robot.move_forward(3)
###
# move back and forth between the chair and the table 3 times
pos1 = robot.get_pos('chair')
pos2 = robot.get_pos('table')
for i in range(3):
    robot.move_to(pos1)
    robot.move_to(pos2)
###
# go to the television
robot.move_to_object('television')
###
# monitor the chair from 4 view points
contour = robot.get_contour('chair')
np.random.shuffle(contour)
for pos in contour[:4]:
    robot.move_to(pos)
    robot.face('chair')
###
# navigate to 3 meters right of the table
robot.move_to_right('table')
robot.face('table')
robot.turn(180)
robot.move_forward(3)
###
# turn west
robot.turn_absolute(-90)
###
# turn east
robot.turn_absolute(90)
###
# turn south
robot.turn_absolute(180)
###
# turn north
robot.turn_absolute(0)
###
# turn east and then turn left 90 degrees
robot.turn_absolute(90)
robot.turn(-90)
###
# move 3 meters north of the chair
robot.move_north('chair')
robot.turn_absolute(0)
robot.move_forward(3)
###
# move 1 meters south of the chair
robot.move_south('chair')
robot.turn_absolute(180)
robot.move_forward(3)
###
# move 3 meters west
robot.turn_absolute(-90)
robot.move_forward(3)
# {lang}
    """
        print("lang: ", lang)
        response=openai_model.chat(question)
        # response = openai.Completion.create(
        #     engine="text-davinci-002",
        #     prompt=question,
        #     max_tokens=400,
        #     temperature=0.0,
        #     stop="###",
        # )
        # result = response["choices"][0]["text"].strip()
        result = response.strip()
        if result:
            results += result + "\n"
        print(result)
    print("generated code")
    print(results)
    return results
