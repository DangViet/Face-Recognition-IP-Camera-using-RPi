#!/usr/bin/env/python
from twilio.rest import TwilioRestClient
 
 
#----------------------------------------------------------------------
def send_sms(msg, to):
    """"""
    sid = "ACd5f1742c704377e4082b743ee8be8365"
    auth_token = "e48f7aca08923333142095421a10c5b3"
    twilio_number = "19165206460"
 
    client = TwilioRestClient(sid, auth_token)
 
    message = client.messages.create(body=msg,
                                     from_=twilio_number,
                                     to=to,
                                     )
 
if __name__ == "__main__":
    msg = "Test Twilio on Raspberry Pi"
    to ="+84 122 513 9439"
    send_sms(msg, to)
