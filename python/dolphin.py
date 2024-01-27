from transformers import pipeline

pipe = pipeline("text-generation", model="TheBloke/MegaDolphin-120b-GPTQ")

result = pipe("Generate a poem about the seasons.")

print(result)