"""Test doubles for generation."""

from src.generation.generator import ProviderError


class FakeProvider:
    """Return scripted responses and record requests."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def complete(self, system, prompt):
        self.calls.append((system, prompt))
        response = next(self.responses)
        if isinstance(response, ProviderError):
            raise response
        return response
