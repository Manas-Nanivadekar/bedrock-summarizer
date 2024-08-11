import json
import os
import sys
import boto3

# Imports for the Bedrock model
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
