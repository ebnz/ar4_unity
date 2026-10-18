import re
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class VariableInfo:
    name: str
    type: str
    description: str


@dataclass
class ObservationInfo:
    name: str
    description: str


@dataclass
class ActionInfo:
    continuous_actions: int
    discrete_actions: int
    description: str


@dataclass
class EnvironmentInfo:
    task_name: str
    task_description: str
    observations_code: str
    action_received_code: str
    success_criteria: str
    variables: List[VariableInfo]
    observations: Dict[str, str]
    actions: ActionInfo
    context: Dict[str, str]

    def to_prompt(self) -> str:
        """Formats the environment information for an LLM prompt."""
        output = []

        output.append(f"Task Description: {self.task_description}")
        output.append(f"Success Criteria: {self.success_criteria}")

        if self.observations:
            output.append("\nAvailable Observations:")
            for obs_name, obs_desc in self.observations.items():
                output.append(f"  - observations['{obs_name}']: {obs_desc}")

        if self.actions:
            output.append(
                f"\nAction Space: {self.actions.continuous_actions} continuous actions"
            )
            output.append(f"Description: {self.actions.description}")

        if self.context:
            output.append("\nContext Variables:")
            for ctx_name, ctx_desc in self.context.items():
                output.append(f"  - context.{ctx_name}: {ctx_desc}")

        if self.variables:
            output.append("\nEnvironment Variables:")
            for var in self.variables:
                output.append(f"  - {var.name} ({var.type}): {var.description}")
            output.append("")

        return "\n".join(output)


class UnityAgentParser:
    def __init__(self, cs_file_path: str):
        self.cs_file_path = cs_file_path
        with open(cs_file_path, "r") as f:
            self.content = f.read()

    def extract_task_name(self) -> str:
        """Extract task name from // @task-name: comment"""
        # Look for a line with // @task-name: <name>
        match = re.search(r"//\s*@task-name:\s*(\S+)", self.content)
        if not match:
            raise ValueError(
                "No // @task-name: comment found in the Unity agent file. Please add a task name."
            )
        return match.group(1)

    def extract_task_description(self) -> str:
        """Extract task description from XML summary block"""
        # Look for /// <summary> block
        summary_match = re.search(
            r"///\s*<taskdescription>\s*(.*?)\s*///\s*</taskdescription>",
            self.content,
            re.DOTALL,
        )

        if not summary_match:
            raise ValueError(
                "No /// <taskdescription> block found in the Unity agent file. Please add a task description."
            )

        # Extract and clean the summary text
        summary_text = summary_match.group(1)
        # Remove /// from each line and clean up
        lines = []
        for line in summary_text.split("\n"):
            cleaned = re.sub(r"^\s*///\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)

        return " ".join(lines)

    def extract_success_criteria(self) -> str:
        """Extract success criteria from XML successcriteria block"""
        # Look for /// <successcriteria> block
        success_criteria_match = re.search(
            r"///\s*<successcriteria>\s*(.*?)\s*///\s*</successcriteria>",
            self.content,
            re.DOTALL,
        )

        if not success_criteria_match:
            raise ValueError(
                "No /// <successcriteria> block found in the Unity agent file. Please add a success criteria."
            )

        # Extract and clean the success criteria text
        success_criteria_text = success_criteria_match.group(1)
        # Remove /// from each line and clean up
        lines = []
        for line in success_criteria_text.split("\n"):
            cleaned = re.sub(r"^\s*///\s*", "", line).strip()
            if cleaned:
                lines.append(cleaned)

        return " ".join(lines)

    def extract_observations(self) -> Tuple[str, Dict[str, str]]:
        """Extract CollectObservations method code and observation descriptions"""
        # Find CollectObservations method
        obs_method_match = re.search(
            r"public override void CollectObservations\(VectorSensor sensor\)\s*{(.*?)^    }",
            self.content,
            re.MULTILINE | re.DOTALL,
        )

        if not obs_method_match:
            raise ValueError(
                "CollectObservations method not found in the Unity agent file."
            )

        method_code = obs_method_match.group(1).strip()

        # Extract observation assignments with @obs: comments
        observations = {}
        lines = method_code.split("\n")

        for line in lines:
            # Look for pattern: currentObservations["ObsName"] = value; // @obs: description
            obs_match = re.search(
                r'currentObservations\["(\w+)"\]\s*=.*?//\s*@obs:\s*(.+?)$', line
            )
            if obs_match:
                obs_name = obs_match.group(1)
                obs_desc = obs_match.group(2).strip()
                observations[obs_name] = obs_desc

        return method_code, observations

    def extract_action_received(self) -> Tuple[str, ActionInfo]:
        """Extract OnActionReceived method code and action info"""
        # Find OnActionReceived method
        action_method_match = re.search(
            r"public override void OnActionReceived\(ActionBuffers actions\)\s*{(.*?)^    }",
            self.content,
            re.MULTILINE | re.DOTALL,
        )

        if not action_method_match:
            raise ValueError(
                "OnActionReceived method not found in the Unity agent file."
            )

        method_code = action_method_match.group(1).strip()

        # Count continuous and discrete actions
        continuous_count = len(
            re.findall(r"actions\.ContinuousActions\[(\d+)\]", method_code)
        )
        discrete_count = len(
            re.findall(r"actions\.DiscreteActions\[(\d+)\]", method_code)
        )

        # Extract action description from method
        description = "Delta movements for target position"
        if "targetDeltaAction" in method_code:
            description = "Delta movements for target position in X, Y, Z axes"

        action_info = ActionInfo(
            continuous_actions=continuous_count,
            discrete_actions=discrete_count,
            description=description,
        )

        return method_code, action_info

    def extract_context_variables(self) -> Dict[str, str]:
        """Extract reward context variables with @context: comments"""
        # Find ExecuteRewardFunction method where context is defined
        # Note: async is optional - some agents use async, some don't
        context_method_match = re.search(
            r"private\s+(?:async\s+)?void ExecuteRewardFunction\(\)\s*{(.*?)^    }",
            self.content,
            re.MULTILINE | re.DOTALL,
        )

        if not context_method_match:
            return {}  # No context found, return empty dict

        method_code = context_method_match.group(1)
        context_vars = {}
        lines = method_code.split("\n")

        for line in lines:
            # Look for pattern: ["contextName"] = value, // @context: description
            context_match = re.search(
                r'\["(\w+)"\]\s*=.*?//\s*@context:\s*(.+?)$', line
            )
            if context_match:
                var_name = context_match.group(1)
                var_desc = context_match.group(2).strip()
                context_vars[var_name] = var_desc

        return context_vars

    def extract_variables_with_summaries(self) -> List[VariableInfo]:
        """Extract public variables that have /// <summary> documentation"""
        variables = []

        # Split content into lines for line-by-line processing
        lines = self.content.split("\n")

        i = 0
        while i < len(lines):
            # Check if this line starts a summary block
            if "/// <summary>" in lines[i]:
                summary_lines = []
                j = i + 1

                # Collect summary content until we hit the closing tag
                while j < len(lines) and "/// </summary>" not in lines[j]:
                    # Extract content after ///
                    cleaned = re.sub(r"^\s*///\s*", "", lines[j]).strip()
                    if cleaned:
                        summary_lines.append(cleaned)
                    j += 1

                # Check if the next non-empty line after </summary> is a public variable
                k = j + 1
                while k < len(lines) and lines[k].strip() == "":
                    k += 1

                if k < len(lines):
                    # Pattern for public variable declaration
                    var_pattern = r"^\s*public\s+(\w+(?:\[\])?)\s+(\w+)\s*[;=]"
                    match = re.match(var_pattern, lines[k])

                    if match:
                        var_type = match.group(1)
                        var_name = match.group(2)
                        description = " ".join(summary_lines)

                        variables.append(
                            VariableInfo(
                                name=var_name, type=var_type, description=description
                            )
                        )

                i = k
            else:
                i += 1

        return variables

    def parse(self) -> EnvironmentInfo:
        """Parse the entire Unity agent file and extract environment information"""
        task_name = self.extract_task_name()
        task_desc = self.extract_task_description()
        observations_code, observations_dict = self.extract_observations()
        action_received_code, action_info = self.extract_action_received()
        success_criteria = self.extract_success_criteria()
        variables = self.extract_variables_with_summaries()
        context_vars = self.extract_context_variables()

        return EnvironmentInfo(
            task_name=task_name,
            task_description=task_desc,
            observations_code=observations_code,
            action_received_code=action_received_code,
            success_criteria=success_criteria,
            variables=variables,
            observations=observations_dict,
            actions=action_info,
            context=context_vars,
        )


if __name__ == "__main__":
    # Example usage
    cs_file_path = "../Scripts/ml/RobotAgentDynamic.cs"

    parser = UnityAgentParser(cs_file_path)
    env_info = parser.parse()

    print("=== Extracted Environment Information ===\n")
    print(env_info.to_prompt())
