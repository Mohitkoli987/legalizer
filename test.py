import google.generativeai as genai
genai.configure(api_key="AIzaSyAIP_FcVIFxRAf6V6boPU8z9_yyU-EbMF0")

for m in genai.list_models():
    print(m.name)
