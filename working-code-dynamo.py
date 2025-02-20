import json
import boto3
import urllib3
import hashlib
import os
import re
import time
import botocore.exceptions  # Needed to catch ClientError

# Initialize clients and resources
bedrock = boto3.client(service_name='bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('AIState')
slackUrl = 'https://slack.com/api/chat.postMessage'
SlackChatHistoryUrl = 'https://slack.com/api/conversations.replies'
slackToken = os.environ.get('token')
http = urllib3.PoolManager()


# --- Claude Invocation with Throttling Handling ---
def call_bedrock(msg):
    """
    Builds a prompt from a list of conversation messages and calls the Claude model via Bedrock.
    Implements exponential backoff for throttling errors.
    """
    prompt = "\n".join(msg)
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.1,
        "top_k": 250,
        "stop_sequences": [],
        "top_p": 0.9,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    body = json.dumps(payload)
    
    # Set up retry parameters
    retries = 0
    max_retries = 5
    backoff = 1  # seconds

    while retries < max_retries:
        try:
            response = bedrock.invoke_model(
                body=body,
                modelId='arn:aws:bedrock:us-east-1:851725293109:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                accept='application/json',
                contentType='application/json'
            )
            break  # If successful, exit the retry loop
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ThrottlingException':
                print(f"ThrottlingException encountered. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2  # exponential backoff
            else:
                # For other errors, re-raise the exception
                raise
    else:
        # If we exit the loop without a successful response, raise an error.
        raise Exception("Max retries exceeded due to throttling.")

    response_body = json.loads(response.get('body').read())
    if ('content' in response_body and 
        isinstance(response_body['content'], list) and 
        len(response_body['content']) > 0):
        completion = response_body['content'][0].get('text', '')
        completion = re.sub(r'^\s*Assistant: ?', '', completion)
        return completion.strip()
    else:
        return ''


# --- Simple Utility Functions for Duplicate-Message Prevention ---
def hash_message(message):
    msg_bytes = message.encode('utf-8')
    sha1 = hashlib.sha1(msg_bytes)
    return sha1.hexdigest()


def get_message_hash(user_id):
    response = table.get_item(Key={'id': user_id})
    if 'Item' in response and 'last_message_hash' in response['Item']:
        return response['Item']['last_message_hash']
    return None


def set_message_hash(user_id, message):
    table.update_item(
        Key={'id': user_id},
        UpdateExpression="SET last_message_hash = :value",
        ExpressionAttributeValues={':value': hash_message(message)}
    )


# --- User Name Memory Functions ---
def get_user_name(user_id):
    response = table.get_item(Key={'id': user_id})
    if 'Item' in response and 'user_name' in response['Item']:
        return response['Item']['user_name']
    return None


def set_user_name(user_id, name):
    table.update_item(
        Key={'id': user_id},
        UpdateExpression="SET user_name = :value",
        ExpressionAttributeValues={':value': name}
    )


# --- Lambda Handler ---
def lambda_handler(event, context):
    headers = {
        'Authorization': f'Bearer {slackToken}',
        'Content-Type': 'application/json',
    }
    slackBody = json.loads(event['body'])
    print(json.dumps(slackBody))
    
    slackEvent = slackBody.get('event', {})
    slackText = slackEvent.get('text', '')
    slackUser = slackEvent.get('user', '')
    channel = slackEvent.get('channel', '')
    thread_ts = slackEvent.get('thread_ts')
    ts = slackEvent.get('ts', '')
    eventType = slackEvent.get('type', '')
    subtype = slackEvent.get('subtype')
    bot_id = slackEvent.get('bot_id')
    
    lower_text = slackText.lower()
    
    # 1. If the user tells the bot their name, e.g. "my name is bujji thalli"
    name_match = re.search(r"my name is\s+([\w\-\_]+)", slackText, re.IGNORECASE)
    if name_match:
        name = name_match.group(1)
        set_user_name(slackUser, name)
        data = {
            'channel': channel,
            'text': f"<@{slackUser}> Got it, I'll remember that your name is {name}.",
            'thread_ts': thread_ts or ts
        }
        http.request('POST', slackUrl, headers=headers, body=json.dumps(data))
        return {
            'statusCode': 200,
            'body': json.dumps({'msg': "Name stored"})
        }
    
    # 2. If the user asks "do you remember my name"
    if "do you remember my name" in lower_text:
        user_name = get_user_name(slackUser)
        if user_name:
            reply = f"Yes, your name is {user_name}."
        else:
            reply = "I don't seem to remember your name yet. What should I call you?"
        data = {
            'channel': channel,
            'text': f"<@{slackUser}> {reply}",
            'thread_ts': thread_ts or ts
        }
        http.request('POST', slackUrl, headers=headers, body=json.dumps(data))
        return {
            'statusCode': 200,
            'body': json.dumps({'msg': "Name query processed"})
        }
    
    # --- Process conversation messages ---
    if get_message_hash(slackUser) == hash_message(slackText):
        # Already processed this message.
        return {
            'statusCode': 200,
            'body': json.dumps({'msg': "Duplicate message; skipping."})
        }
    
    # For thread replies: build conversation history.
    if eventType == 'message' and bot_id is None and subtype is None and thread_ts is not None:
        set_message_hash(slackUser, slackText)
        bedrockMsg = []
        is_last_message_from_bot = False
        
        # Retrieve the full thread history from Slack.
        historyResp = http.request('GET', f"{SlackChatHistoryUrl}?channel={channel}&ts={thread_ts}", headers=headers)
        # Decode the response and load the JSON.
        history_data = json.loads(historyResp.data.decode('utf-8'))
        messages = history_data.get('messages', [])
        
        for message in messages:
            cleanMsg = re.sub(r'<@.*?>', '', message.get('text', ''))
            bot_profile = message.get('bot_profile')
            if bot_profile is None:
                bedrockMsg.append(f'Human: {cleanMsg}')
                is_last_message_from_bot = False
            else:
                bedrockMsg.append(f'\n\nAssistant: {cleanMsg}')
                is_last_message_from_bot = True
        
        bedrockMsg.append('\n\nAssistant:')
 
        if not is_last_message_from_bot:
            msg = call_bedrock(bedrockMsg)
            data = {
                'channel': channel,
                'text': f"<@{slackUser}> {msg}",
                'thread_ts': thread_ts
            }
            http.request('POST', slackUrl, headers=headers, body=json.dumps(data))
    
    # For direct app mentions (outside of threads).
    elif eventType == 'app_mention' and bot_id is None and thread_ts is None:
        initMsg = re.sub(r'<@.*?>', '', slackText)
        bedrockMsg = [f'Human: {initMsg} \n\nAssistant:']
        msg = call_bedrock(bedrockMsg)
        data = {
            'channel': channel,
            'text': f"<@{slackUser}> {msg}",
            'thread_ts': ts
        }
        http.request('POST', slackUrl, headers=headers, body=json.dumps(data))
    
    return {
        'statusCode': 200,
        'body': json.dumps({'msg': "message received"})
    }
