import os
from simple_salesforce import Salesforce
sf = Salesforce(username='animesh.das.myorg2@gmail.com', password='Matlab@00000', security_token='ukZpE0RkPLSYongDC85NF69c')

print(sf.query("Select Id, Name, rule_type__c, tier_number__c FROM QGenix_LLM_Param__c"))