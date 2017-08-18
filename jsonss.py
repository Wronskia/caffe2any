import json

dic={"conv1": "GEMMS","conv2": "GEMMS","conv3": "GEMMS","conv4": "GEMMS","conv5": "GEMMS"}
with open('test11.json','w') as file:
		print("ksssssss")
		json.dump({"conv1": "GEMMS","conv2": "GEMMS","conv3": "GEMMS","conv4": "GEMMS","conv5": "GEMMS"},file)


with open('test11.json','r') as json_data:
		config_json_param = json.load(json_data)
