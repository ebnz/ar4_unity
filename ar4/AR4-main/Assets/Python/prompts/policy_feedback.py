policy_feedback_prompt = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components
in the reward function as well as global policy metrics such as success rates and episode lengths after every
{checkpoint_interval} steps and the maximum, mean, minimum values encountered. Please provide an analysis of
the policy performance and suggest improvements. Don't write a new function yet, just analyze the current one.
"""
