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
template_strin_2 = """{%- set shared_prompt = __RULE__ %}
{{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>' -}}
{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
<think>

</think>
{%- endif %}""".replace("__RULE__", repr(SHARED_RPG_RULE))

# 3. Compile the template using Jinja2
template_strings = [template_string_1, template_strin_2]
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