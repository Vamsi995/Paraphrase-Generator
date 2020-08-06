import requests
t = requests.post("https://service.afterthedeadline.com/checkDocument", data=["this is it"])

print(t.text)
