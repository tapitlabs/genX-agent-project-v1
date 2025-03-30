import os
import json
from simple_salesforce import Salesforce
# Set SFDC API credentials
SFDC_USERNAME = os.environ["SFDC_USERNAME"]
SFDC_USERPWD = os.environ["SFDC_USERPWD"]
SFDC_TOKEN = os.environ["SFDC_TOKEN"]
sf = Salesforce(username=SFDC_USERNAME, password=SFDC_USERPWD, security_token=SFDC_TOKEN)

def get_LLM_Param():
    result = ""
    records = sf.query("Select Id, Name, rule_type__c, tier_number__c FROM QGenix_LLM_Param__c")
    res = json.loads(json.dumps(records))
    for r in res["records"]:
        result = result + '\n' + r["Name"]
    return result