##Environment Creation
conda create --name <env> --file requirements.txt

##Souce Code Information
e_commerce_dialogue_flow.py --> For Dialogue Flow Generation

promptv5.py --> For Dialogue generation with the help of generated dialogue flow by e_commerce_dialogue_flow.py

Background Dataset --> product_data.json

###To Generate dataset:
run e_commerce_dialogue_flow.py
then
run promptv5.py

Integrative Negotiation Data --> IND.csv

To run INA agent in INA folder
To train the agent and customer model both run INA_main.py
To test the the agent and customer run interact_nego.py

Similary for baseline model go to ARDM
train the model with ardm.py
to test the model run interact.py

similarly for other models.


