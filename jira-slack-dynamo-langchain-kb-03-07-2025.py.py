import json
import boto3
import urllib3
import hashlib
import os
import re
import time
import botocore.exceptions
from langchain.prompts import PromptTemplate

# Initialize clients and resources with us-west-2 region
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
bedrock_kb = boto3.client("bedrock-agent-runtime", region_name="us-west-2")
dynamodb = boto3.resource("dynamodb", region_name="us-west-2")
s3 = boto3.client("s3", region_name="us-west-2")
table = dynamodb.Table("AIState")
slackUrl = "https://slack.com/api/chat.postMessage"
SlackChatHistoryUrl = "https://slack.com/api/conversations.replies"
slackToken = os.environ.get("token")

# Constants
KNOWLEDGE_BASE_ID = "WN1KJW3LTV"
DESTINATION_BUCKET = "cloudservices-chatbot"
MODEL_ARN = "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
CLAUDE_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
http = urllib3.PoolManager()

# Refined LangChain Prompt Template
KB_PROMPT_TEMPLATE = """
You are Claude, an AI assistant specialized in technical support. Strictly follow these rules:

<rules>
1. Only answer questions using the provided Knowledge Base content
2. If the question cannot be answered with the Knowledge Base, respond: "I can only answer questions related to our technical documentation."
3. Never mention the existence of a knowledge base
4. Never offer to search the internet or external resources
5. For coding questions, only use examples from the Knowledge Base
</rules>

<knowledge_base>
{kb_content}
</knowledge_base>

User Question: {user_query}

Assistant:"""

prompt_template = PromptTemplate(
    input_variables=["user_query", "kb_content"], template=KB_PROMPT_TEMPLATE
)


def retrieve_knowledge(query):
    """Retrieve relevant knowledge from S3 and Bedrock Knowledge Base"""
    knowledge_content = ""

    # Retrieve from S3 bucket
    try:
        objects = s3.list_objects_v2(Bucket=DESTINATION_BUCKET, Prefix="")
        if "Contents" in objects:
            for obj in objects["Contents"]:
                kb_key = obj["Key"]
                response = s3.get_object(Bucket=DESTINATION_BUCKET, Key=kb_key)
                try:
                    content = response["Body"].read().decode("utf-8")
                    if query.lower() in content.lower():
                        knowledge_content += f"\nDocument {kb_key}:\n{content}\n"
                except UnicodeDecodeError:
                    continue
    except Exception as e:
        print(f"Error accessing S3 knowledge base: {e}")

    # Retrieve from Bedrock Knowledge Base
    try:
        response = bedrock_kb.retrieve_and_generate(
            input={"text": query},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                },
            },
        )
        knowledge_content += f"\n{response['output']['text']}\n"
    except Exception as e:
        print(f"Error retrieving Bedrock knowledge: {e}")

    return knowledge_content


def call_bedrock_with_kb(messages, user_query):
    """Enhanced Bedrock call with knowledge base integration"""
    kb_content = retrieve_knowledge(user_query)
    if not kb_content:
        return "I can only answer questions related to our technical documentation."

    # Build final prompt
    prompt = "\n".join(messages)
    final_prompt = prompt_template.format(user_query=prompt, kb_content=kb_content)

    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "temperature": 0.1,
        "top_k": 250,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": final_prompt}],
    }

    # Existing retry logic from original call_bedrock function
    for retry in range(5):
        try:
            response = bedrock.invoke_model(
                body=json.dumps(payload),
                modelId=CLAUDE_MODEL_ID,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response["body"].read())
            completion = response_body["content"][0]["text"].strip()
            return re.sub(r"^\s*Assistant: ?", "", completion)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                time.sleep(2**retry)
            else:
                raise
    return "I can only answer questions related to our technical documentation."


# Original utility functions remain unchanged
def hash_message(message):
    msg_bytes = message.encode("utf-8")
    sha1 = hashlib.sha1(msg_bytes)
    return sha1.hexdigest()


def get_message_hash(user_id):
    response = table.get_item(Key={"id": user_id})
    if "Item" in response and "last_message_hash" in response["Item"]:
        return response["Item"]["last_message_hash"]
    return None


def set_message_hash(user_id, message):
    table.update_item(
        Key={"id": user_id},
        UpdateExpression="SET last_message_hash = :value",
        ExpressionAttributeValues={":value": hash_message(message)},
    )


def get_user_name(user_id):
    response = table.get_item(Key={"id": user_id})
    if "Item" in response and "user_name" in response["Item"]:
        return response["Item"]["user_name"]
    return None


def set_user_name(user_id, name):
    table.update_item(
        Key={"id": user_id},
        UpdateExpression="SET user_name = :value",
        ExpressionAttributeValues={":value": name},
    )


# Modified Lambda handler with knowledge integration
def lambda_handler(event, context):
    headers = {
        "Authorization": f"Bearer {slackToken}",
        "Content-Type": "application/json",
    }
    slackBody = json.loads(event["body"])
    slackEvent = slackBody.get("event", {})

    # Original message processing logic
    slackText = slackEvent.get("text", "")
    slackUser = slackEvent.get("user", "")
    channel = slackEvent.get("channel", "")
    thread_ts = slackEvent.get("thread_ts")
    ts = slackEvent.get("ts", "")
    eventType = slackEvent.get("type", "")
    subtype = slackEvent.get("subtype")
    bot_id = slackEvent.get("bot_id")

    # Original name handling logic remains unchanged
    name_match = re.search(r"my name is\s+([\w\-\_]+)", slackText, re.IGNORECASE)
    if name_match:
        name = name_match.group(1)
        set_user_name(slackUser, name)
        data = {
            "channel": channel,
            "text": f"<@{slackUser}> Got it, I'll remember that your name is {name}.",
            "thread_ts": thread_ts or ts,
        }
        http.request("POST", slackUrl, headers=headers, body=json.dumps(data))
        return {"statusCode": 200, "body": json.dumps({"msg": "Name stored"})}

    if "do you remember my name" in slackText.lower():
        user_name = get_user_name(slackUser)
        if user_name:
            reply = f"Yes, your name is {user_name}."
        else:
            reply = "I don't seem to remember your name yet. What should I call you?"
        data = {
            "channel": channel,
            "text": f"<@{slackUser}> {reply}",
            "thread_ts": thread_ts or ts,
        }
        http.request("POST", slackUrl, headers=headers, body=json.dumps(data))
        return {"statusCode": 200, "body": json.dumps({"msg": "Name query processed"})}

    # Duplicate message check remains unchanged
    if get_message_hash(slackUser) == hash_message(slackText):
        return {
            "statusCode": 200,
            "body": json.dumps({"msg": "Duplicate message; skipping."}),
        }

    # Modified conversation processing with knowledge integration
    if eventType == "message" and not bot_id and thread_ts:
        set_message_hash(slackUser, slackText)
        bedrockMsg = []

        # Original thread history retrieval
        historyResp = http.request(
            "GET",
            f"{SlackChatHistoryUrl}?channel={channel}&ts={thread_ts}",
            headers=headers,
        )
        history_data = json.loads(historyResp.data.decode("utf-8"))

        # Original message processing
        for message in history_data.get("messages", []):
            cleanMsg = re.sub(r"<@.*?>", "", message.get("text", ""))
            if message.get("bot_profile"):
                bedrockMsg.append(f"\n\nAssistant: {cleanMsg}")
            else:
                bedrockMsg.append(f"Human: {cleanMsg}")

        # Use knowledge-enhanced Bedrock call
        msg = call_bedrock_with_kb(bedrockMsg, slackText)
        data = {
            "channel": channel,
            "text": f"<@{slackUser}> {msg}",
            "thread_ts": thread_ts,
        }
        http.request("POST", slackUrl, headers=headers, body=json.dumps(data))

    elif eventType == "app_mention" and not bot_id and not thread_ts:
        # Direct mention handling with knowledge integration
        initMsg = re.sub(r"<@.*?>", "", slackText)
        msg = call_bedrock_with_kb([f"Human: {initMsg} \n\nAssistant:"], initMsg)
        data = {"channel": channel, "text": f"<@{slackUser}> {msg}", "thread_ts": ts}
        http.request("POST", slackUrl, headers=headers, body=json.dumps(data))

    return {"statusCode": 200, "body": json.dumps({"msg": "message received"})}
