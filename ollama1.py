import ollama
import re


# generate
resp = ollama.generate(model="llama3.2.1b",prompt="why is the plant leaves green in color?")
# print(resp)
# print(type(resp))
# print(resp.model_dump().keys())
# print(resp.response)
resp_text = resp.response
print(resp_text)


#for thinking models
resp1 = ollama.generate(model="qwen3:8b",prompt="Why is the plam green in color?")
resp1_text = resp1.response
act_resp1 = re.sub(r"<think>.*?</think>","",resp1_text,flags=re.DOTALL).strip()
print(act_resp1)

#stream parameters
stream = ollama.generate(model="llama3.2:1b",prompt="why is plant leaves green in color", stream=True)
print(stream)

for i in stream:
    print(i)
    print("**")

for i in stream:
    print(i["response"],end="")


#multimodel input
import base64

image_path = "mine.png"
with open(image_path,"rb") as f:
    image_bytes = f.read()
image_64 = base64.b64encode(image_bytes).decode("utf-8")

img_resp = ollama.generate(model="llava:7b",images=[image_64],prompt="Describe the image")
print(img_resp.response)


#multiple images to input
image_paths = ["mine.png","mine1.png","mine2.png"]

images_base64 = []
for i in image_paths:
    with open(image_paths,"rb") as f:
        images_bytes = f.read()
    images_base64.append(base64.b64encode(images_bytes).decode("utf-8")) 

img1_resp = ollama.generate(model="llava:7b",images=images_base64,prompt="Describe the image")
print(img1_resp.response)


#any structures form like json and etc

paragraph = """A paragraph is a distinct section of writing that focuses on a single idea or topic, consisting of one or more sentences. It is typically marked by an indentation or a skipped line, and its purpose is to organize written work, making it easier for readers to follow the author's thoughts and the structure of the overall text. A strong paragraph often begins with a topic sentence, followed by supporting sentences that provide evidence, examples, and elaboration for the main point."""

ollama.generate(
    model="qwen3:8b",
    prompt=f"Extract the main things from the paragraph and summarise it:\n{paragraph}",
    format={
        #define a json structure by urself
        "type":"object",
        "properties":{
            "people":{

            }
        }
    }
    )


#pass system instruction 
img1_resp = ollama.generate(model="llava:7b",images=images_base64,prompt="Describe the image",system="You are funny in responding")
print(img1_resp.response)


#options
img1_resp = ollama.generate(model="llava:7b",images=images_base64,prompt="Describe the image",options={
    "temperature": 0.7,
    "top_p": 0.5,
})
print(img1_resp.response)