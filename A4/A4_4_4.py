!pip install openai

import os
import openai
openai.api_key = "my parivate key"

completion = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "Determine if the following statement is strong statement or softened statement. If it is strong statement output 0. If it is softened statement output 1. The following provides you with an example of a label 0 statement and a label 1 statement. Label 0 statement: 'You are concerned with the costs of smoking.' Label 1 statement: 'It sounds like the financial aspect of your smoking habit has been catching your attention recently. Can you share a bit more about that with me?' "},
    {"role": "user", "content": "You realize that cigarettes can potentially cause financial stress and you wish you could spend less on tobacco."}
  ]
)

print(completion.choices[0].message)