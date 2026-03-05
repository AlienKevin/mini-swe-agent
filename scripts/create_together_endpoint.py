"""Create a Together AI Dedicated Endpoint for Kimi K2.5."""

import os
from dotenv import load_dotenv
from together import Together

load_dotenv()

client = Together()

response = client.endpoints.create(
    model="moonshotai/Kimi-K2.5",
    display_name="kevinli020508@gmail.com/moonshotai/Kimi-K2.5",
    hardware="8x_nvidia_h200_140gb_sxm",
    autoscaling={
        "min_replicas": 1,
        "max_replicas": 1,
    },
)
print(response)
