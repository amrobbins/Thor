import os

ALL_INTEGRATION_TESTS_ENV = "THOR_ALL_INTEGRATION_TESTS"


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def integration_flag_enabled(name: str) -> bool:
    return env_flag_enabled(ALL_INTEGRATION_TESTS_ENV) or env_flag_enabled(name)


def integration_skip_reason(*specific_envs: str, description: str) -> str:
    if len(specific_envs) == 1:
        return f"set {specific_envs[0]}=1 or {ALL_INTEGRATION_TESTS_ENV}=1 to run {description}"
    joined_envs = " / ".join(f"{env}=1" for env in specific_envs)
    return f"set one of {joined_envs}, or {ALL_INTEGRATION_TESTS_ENV}=1, to run {description}"
