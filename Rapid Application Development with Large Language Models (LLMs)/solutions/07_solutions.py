## The following solution worked well for us

########################################################################################
## Task 2

agent.agent.llm_chain.prompt.template = """
<s>[INST/]<<SYS>>
Answer the following questions as best you can. You have access to the following tools:

[Python REPL]: A Python shell. Use this to execute python commands. Input should be a valid python command.
        If you expect output it should be printed out.

===================================
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of ["Python REPL"]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

===================================
Use the following as an example of how you should perform: 

Question: Please tell me 42 x 24
Thought: I can just use python to perform this operation
Action: Python REPL
Action-Input: 
def multiply_add(a, b, c):
    return a * b + c
print('42 x 24 + 12 is', multiply_add(42, 24, 12))
Observation: 42 x 24 + 12 is 1020
Thought: Based on the observation, I am ready to answer
Final Answer: the final answer is 1008

<</SYS>>
=================================
Question: {input}
[/INST]
Thought: {agent_scratchpad}
"""