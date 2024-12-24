import json
from random import randint
import time
from openai import OpenAI

def create_batch(api_key, prompts, model_name, output_file_name):
    # Initialize the client
    client = OpenAI(
        base_url="https://api.saas.parasail.io/v1",
        api_key=api_key,
    )

    # Create input file for batch
    input_file_name = "batch_input_{}.jsonl".format(randint(10000,99999))
    with open(input_file_name, "w") as file:
        for i, prompt in enumerate(prompts, start=1):
            file.write(
                json.dumps(
                    {
                        "custom_id": f"request-{i}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_name,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    }
                )
                + "\n"
            )

    # Upload input file
    input_file = client.files.create(file=open(input_file_name, "rb"), purpose="batch")

    # Create batch
    batch = client.batches.create(
        input_file_id=input_file.id,
        completion_window="24h",
        endpoint="/v1/chat/completions",
    )

    print(f"Batch {batch.id} created. Status: {batch.status}")

    # Poll for batch completion
    while batch.status != "completed":
        time.sleep(120)
        batch = client.batches.retrieve(batch.id)
        print(f"Status of {batch.id}: {batch.status}")

    # Download output file
    output = client.files.content(batch.output_file_id).content

    with open(output_file_name, "wb") as file:
        file.write(output)

    print(f"Batch output saved to {output_file_name}")
