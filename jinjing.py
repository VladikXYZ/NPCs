from jinja2 import Template

# 1. Your shared rule
SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse."

template_string_1 = """
{%- set shared_prompt = __RULE__ -%}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<|im_start|>system\\n' + messages[0].content + '\\n\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- else -%}
    {{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE))

# 2. Your template (using your exact f-string setup)
template_string_2 = """{%- set shared_prompt = __RULE__ -%}
{{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE))
# 3. Compile the template using Jinja2
template_strings = [template_string_1, template_string_2]
outputs = []

for temp_string in template_strings:
    template = Template(temp_string)

    # 4. Your list of dictionaries (simulating chat history)
    chat_history = [
        {"role": "user", "content": "Hello there, traveler!"},
        {"role": "assistant", "content": "Greetings. What do you seek?"},
        {"role": "user", "content": "Where is the nearest tavern?"}
    ]

    # 5. Render the template by passing the list to the 'messages' variable
    result = template.render(messages=chat_history)

    # Print the final compiled string
    print("--- RENDERED OUTPUT ---")
    print(result)
    outputs.append(result)

string1 = outputs[0]
string2 = outputs[1]
print(repr(string1[-1]))
print(repr(string2[-1]))

# Loop through the length of the shorter string
for i in range(min(len(string1), len(string2))):
    if string1[i] != string2[i]:
        print(f"Strings diverge at index {i}:")
        print(f"  String 1 has '{string1[min(0,i-10):i+10]}'")
        print(f"  String 2 has '{string2[min(0,i-10):i+10]}'")
        break  # Stop at the first mismatch

# Check if one string is just a longer version of the other
if string1.startswith(string2) or string2.startswith(string1):
    if len(string1) != len(string2):
        print(len(string1))
        print(len(string2))
        print(f"Strings match up to index {min(len(string1), len(string2))}, but have different lengths.")