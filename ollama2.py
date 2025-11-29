import ollama
import base64

#does not store any history
while True:
    user_input = input("You: ")

    if user_input.lower()=="quit":
        print("Model: goodbye")
        break

    resp = ollama.generate(model="llama3.2:1b",prompt=user_input)
    print("Model:",resp.response)


# stores history of chat
msg = []
msg.append({"role": "system", "content": "You are a funny assistant"})

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Model: goodbye")
        break

    msg.append({"role": "user", "content": user_input})

    resp = ollama.chat(model="llama3.2:1b", messages=msg)

    reply = resp["message"]["content"]
    print("Model:", reply)

    msg.append({"role": "assistant", "content": reply})

print(msg)






# PASSING IMAGE IN CHAT FUNCTION
image_path= "img.png"

with open(image_path,"rb") as f:
    image_bytes= f.read()
image_64= base64.b64encode(image_bytes).decode("utf-8")


messages=[]
messages.append({"role":"system","content":"YOu are  a funny assistant"})

# lets attach the image
messages.append(
    {"role":"user",
     "content":"Here is the image , I want to talk about",
     "images":[image_64]}
)


while True:
    user_input = input("You: ")

    if user_input.lower()=="quit":
            print("Assistant: Goodbye")
            break

    messages.append({"role":"user", "content":user_input})

    response= ollama.chat(model="llava:7b",messages=messages)
    print("Assistant:",response["message"]["content"])

    messages.append({"role":"assistant","content":response["message"]["content"]})


print(messages)



# OPTIONS PARAMETERS USING OLLAMA
messages=[{"role":"user","content":"Tell me short story about dragons"}]
response=ollama.chat(
    model="llama3.2:1b",
    messages=messages,
    options={
        "temperature":1.0,
        "top_p":0.9,
        "num_predict":100,
        "repeat_penalty":1.2
    }
)
print(response["message"]["content"])





#BASICS OLLAMA COMMANDS
#OLLAMA LIST
local_models= ollama.list()
print(local_models)


for i in local_models["models"]:
    print(i["model"])
    print(i["size"])


#OLLAMA PULL
model_name="deepseek-r1"
progess = ollama.pull(model_name, stream=True)

for i in progess:
    print(i)


#OLLAMA SHOW
models_details = ollama.show("llama3.2:1b")
print(models_details)

model_dict= models_details.dict()
print(model_dict["capabilities"])
print(model_dict["parameters"])



#OLLAMA DELETE
ollama.delete("embeddinggemma:latest")