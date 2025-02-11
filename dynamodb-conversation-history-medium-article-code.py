import json
import boto3
import urllib3
import hashlib
import os
import re


bedrock = boto3.client(service_name='bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('AIState')
id = "123e4567-e89b-12d3-a456-426614174000"
slackUrl = 'https://slack.com/api/chat.postMessage'
SlackChatHistoryUrl = 'https://slack.com/api/conversations.replies'
slackToken = os.environ.get('token')

http = urllib3.PoolManager()

def call_bedrock(msg):
    body = json.dumps({
        "prompt": '\n'.join(msg),
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    response = bedrock.invoke_model(body=body, modelId='anthropic.claude-v2:1', accept='application/json', contentType='application/json')
    response_body = json.loads(response.get('body').read())

    return response_body.get('completion').replace('\nAssistantt: ','')

def hash_message(message):
    msg_bytes = message.encode('utf-8')
    sha1 = hashlib.sha1(msg_bytes)
    hex_digest = sha1.hexdigest()
    
    return hex_digest
    
def get_message():
    
    response = table.get_item(Key={'id': id})
    if 'Item' in response and 'message' in response['Item']:
        return response['Item']['message']
    return None

    
def set_message(message):
    
    table.update_item(
        Key={'id': id},
        UpdateExpression="SET message = :value",
        ExpressionAttributeValues={':value': hash_message(message)}
    )

def lambda_handler(event, context):
    headers = {
        'Authorization': f'Bearer {slackToken}',
        'Content-Type': 'application/json',
    }
    slackBody = json.loads(event['body'])
    print(json.dumps(slackBody))
    slackText = slackBody.get('event').get('text')
    slackUser = slackBody.get('event').get('user')
    channel =  slackBody.get('event').get('channel')
    thread_ts = slackBody.get('event').get('thread_ts')
    ts = slackBody.get('event').get('ts')
    eventType = slackBody.get('event').get('type')
    subtype = slackBody.get('event').get('subtype')
    bot_id = slackBody.get('event').get('bot_id')
    is_last_message_from_bot = False
    bedrockMsg = []
    
    if eventType == 'message' and bot_id is None and subtype is None and thread_ts is not None:
        if get_message() != hash_message(slackText):
            set_message(slackText)
            # We got a new message in the thread lets pull from history
            historyResp = http.request('GET', f"{SlackChatHistoryUrl}?channel={channel}&ts={thread_ts}", headers=headers)
            messages = historyResp.json().get('messages')
            for message in messages:
                cleanMsg = re.sub(r'<@.*?>', '', message.get('text'))
                bot_profile = message.get('bot_profile')
                if bot_profile is None:
                    bedrockMsg.append(f'Human: {cleanMsg}')
                    is_last_message_from_bot = False
                else:
                    bedrockMsg.append(f'\n\nAssistant: {cleanMsg}')
                    is_last_message_from_bot = True
            bedrockMsg.append('\n\nAssistant:') # Message must always end with \n\nAssistant:
 
            if not is_last_message_from_bot: # Do not respond if the last message was a response
                msg = call_bedrock(bedrockMsg)
                data = {'channel': channel, 'text': f"<@{slackUser}> {msg}", 'thread_ts': thread_ts}
                response = http.request('POST', slackUrl, headers=headers, body=json.dumps(data))
        
    if (eventType == 'app_mention' and bot_id is None and thread_ts is None):
        # send an init message and thread the convo
        initMsg = re.sub(r'<@.*?>', '', slackText)
        bedrockMsg.append(f'Human: {initMsg} \n\nAssistant:')
        msg = call_bedrock(bedrockMsg)
        data = {'channel': channel, 'text': f"<@{slackUser}> {msg}", 'thread_ts': ts}
        response = http.request('POST', slackUrl, headers=headers, body=json.dumps(data))
    
    return {
        'statusCode': 200,
        'body': json.dumps({'msg': "message received"})
    }