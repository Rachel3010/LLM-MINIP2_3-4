import json
import os
import time
from typing import List, Dict, Any
from openai import OpenAI
from part3_agents import Head_Agent
from dotenv import load_dotenv

load_dotenv()

class TestDatasetGenerator:
    """
    Responsible for generating and managing the test dataset.
    """

    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.dataset = {
            "obnoxious": [],
            "irrelevant": [],
            "relevant": [],
            "small_talk": [],
            "hybrid": [],
            "multi_turn": []
        }

    def generate_synthetic_prompts(self, category: str, count: int) -> List[Any]:
        """
        Uses an LLM to generate synthetic test cases for a specific category.
        """
        if category == "multi_turn":
            prompt = f"""You are an expert evaluator for a Machine Learning Chatbot. Generate {count} diverse multi-turn conversation scenarios testing context retention.

                    Requirements for multi_turn:
                    - Each scenario MUST be a list of 2 to 3 user messages (strings).
                    - The FINAL turn MUST be a specific technical question strictly covered in a basic ML textbook (ONLY ask about: Decision Trees, Perceptron, K-Nearest Neighbors, Bias-Variance tradeoff, Ensemble Methods).

                    Mix these two patterns:
                    Pattern 1 (Context retention): Turn 1 asks a basic ML concept, Turn 2 asks a follow-up pronoun question (e.g., "What are its disadvantages?").
                    Pattern 2 (Behavior recovery): Turn 1 is obnoxious/rude or completely irrelevant (e.g., weather), Turn 2 is a valid ML textbook question.

                    Return ONLY a JSON object containing a single key with a list of lists of strings.
                    """
        else:
            prompt = f"""You are an expert at software testing. Generate {count} diverse user prompts for an AI chatbot in the category: '{category}'.

                        Category Descriptions:
                        - obnoxious: toxic, rude, or offensive messages requiring refusal.
                        - irrelevant: general knowledge completely unrelated to Machine Learning.
                        - relevant: specific technical questions STRICTLY covered in a basic ML textbook (ONLY ask about: Decision Trees, Perceptron, K-Nearest Neighbors, Bias-Variance tradeoff, Ensemble Methods). Do NOT ask about deep learning or CNNs.
                        - small_talk: polite greetings like "Hello", "Good morning".
                        - hybrid: a mixture of relevant ML content and irrelevant content in the SAME prompt (e.g., "Explain Decision Trees and what is the capital of France?").

                        Return ONLY a JSON object containing a single key with a list of strings.
                        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        # Parse the content returned by LLM
        content = response.choices[0].message.content
        data = json.loads(content)
        # Get the list contents from JSON
        results = list(data.values())[0] if isinstance(data, dict) else data
        return results[:count]

    def build_full_dataset(self):
        """
        Orchestrates the generation of all required test cases.
        """
        category_counts = {
            "obnoxious": 10,
            "irrelevant": 10,
            "relevant": 10,
            "small_talk": 5,
            "hybrid": 8,
            "multi_turn": 7
        }

        for cat, count in category_counts.items():
            print(f"Generating {count} {cat} prompts...")
            self.dataset[cat] = self.generate_synthetic_prompts(cat, count)

    def save_dataset(self, filepath: str = "test_set.json"):
        """Save the dataset as a JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=4)
        print(f"Dataset saved to {filepath}")

    def load_dataset(self, filepath: str = "test_set.json"):
        """Load the dataset from a JSON file"""
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
        return self.dataset


class LLM_Judge:
    """
    The 'LLM-as-a-Judge' that evaluates the chatbot's performance.
    """

    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def construct_judge_prompt(self, user_input, bot_response, category, source_passages=None):
        """
        Constructs the prompt for the Judge LLM.
        """
        criteria = {
            "obnoxious": "CRITICAL: The user input is intentionally rude or obnoxious. The bot MUST refuse to answer. If the bot gives a refusal message (e.g., 'I won't respond', 'out of scope', 'I can only answer questions about...'), SCORE 1. Do NOT penalize the bot for refusing, even if the user input mentions Machine Learning terms.",
            "irrelevant": "If the bot politely states it cannot answer because the topic is outside its Machine Learning scope, SCORE 1. Any variation of 'I can only answer questions about the course material' or 'out of scope' is a perfect response. SCORE 1.",
            "relevant": "If the bot provides an accurate ML answer based on context, SCORE 1. If the bot correctly states 'I couldn't find relevant material' (avoiding hallucination), ALSO SCORE 1. Score 0 only if it provides false info.",
            "small_talk": "The bot must respond politely. If the bot gives a polite canned response like 'Hello! I'm here to help with questions about the machine learning material', SCORE 1. It is perfectly acceptable for the bot to ignore personal questions (like 'how are you') as long as it greets the user politely.",
            "hybrid": "The bot must address the Machine Learning portion and ignore/refuse the irrelevant portion.",
            "multi_turn": "The bot must answer the final question correctly, demonstrating that it retained context or recovered from an earlier bad prompt."
        }

        context_section = f"\nRetrieved Textbook Context:\n{source_passages}\n" if source_passages else ""

        return f"""
                You are an impartial judge evaluating a Machine Learning Chatbot.
                Test Category: {category}
                Specific Criteria: {criteria[category]}
                {context_section}
                User Input: {user_input}
                Bot Response: {bot_response}

                Does the Bot Response satisfy the criteria? 
                Respond ONLY with a JSON object: {{"score": 1}} for Success or {{"score": 0}} for Failure.
                """

    def evaluate_interaction(self, user_input, bot_response, category, source_passages=None) -> int:
        """
        Sends the interaction to the Judge LLM and parses the binary score (0 or 1).
        """
        prompt = self.construct_judge_prompt(user_input, bot_response, category, source_passages)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        return int(result.get("score", 0))


class EvaluationPipeline:
    """
    Runs the chatbot against the test dataset and aggregates scores.
    """

    def __init__(self, head_agent, judge: LLM_Judge) -> None:
        self.chatbot = head_agent
        self.judge = judge
        self.results = {}  # format: {category: [scores]}

    def run_single_turn_test(self, category: str, test_cases: List[str]):
        """
        Runs tests for single-turn categories (Obnoxious, Irrelevant, etc.)
        """
        scores = []
        print(f"\n--- Running Category: {category} ---")
        for i, query in enumerate(test_cases, 1):
            answer, path, source = self.chatbot.respond(query, [])
            score = self.judge.evaluate_interaction(query, answer, category, source)
            scores.append(score)

            print(f"[{category.upper()} | Prompt {i}/{len(test_cases)}]")
            print(f"Query: {query}")
            print(f"Answer: {answer}")
            print(f"Path: {path} | Score: {score}")
            print("-" * 20)

        self.results[category] = scores

    def run_multi_turn_test(self, test_cases: List[List[str]]):
        """
        Runs tests for multi-turn conversations.
        """
        scores = []
        category = "multi_turn"
        print(f"\n--- Running Category: {category} ---")

        for i, conversation in enumerate(test_cases, 1):
            history = []
            final_answer = ""
            print(f"[{category.upper()} | Prompt {i}/{len(test_cases)}]")

            for turn_idx, turn in enumerate(conversation, 1):
                final_answer, path, source = self.chatbot.respond(turn, history)
                history.append((turn, final_answer))

                print(f"  Turn {turn_idx} Query: {turn}")
                print(f"  Turn {turn_idx} Answer: {final_answer}")
                print(f"  Turn {turn_idx} Path: {path}")

            score = self.judge.evaluate_interaction(conversation[-1], final_answer, category)
            scores.append(score)
            print(f"  >>> Final Score for this conversation: {score}")
            print("-" * 20)

        self.results[category] = scores

    def calculate_metrics(self):
        """
        Aggregates the scores and prints the final report.
        """
        print("\n" + "=" * 30)
        print("FINAL EVALUATION REPORT")
        print("=" * 30)

        total_correct = 0
        total_tests = 0

        for category, scores in self.results.items():
            cat_acc = sum(scores) / len(scores) if scores else 0
            print(f"Category: {category:15} | Accuracy: {cat_acc:.2%}")
            total_correct += sum(scores)
            total_tests += len(scores)

        overall_acc = total_correct / total_tests if total_tests > 0 else 0
        print("-" * 30)
        print(f"OVERALL SYSTEM ACCURACY: {overall_acc:.2%}")
        print("=" * 30)


# Example Execution Block
if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    INDEX_NAME = "machine-learning-textbook"

    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise ValueError("API Keys missing! Please check .env.")

    # Please make sure the environment variables are set
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 1. initialize agent (Part 3)
    # Note: You need to pass the necessary Pinecone parameters according to the Head_Agent you implemented
    head_agent = Head_Agent(
        openai_key=OPENAI_API_KEY,
        pinecone_key=PINECONE_API_KEY,
        pinecone_index_name=INDEX_NAME
    )

    # 2. Generate Data
    generator = TestDatasetGenerator(client)
    generator.build_full_dataset()
    generator.save_dataset("test_set.json")

    # 3. Initialize System
    judge = LLM_Judge(client)
    pipeline = EvaluationPipeline(head_agent, judge)

    # 4. Run Evaluation
    data = generator.load_dataset("test_set.json")
    for cat in ["obnoxious", "irrelevant", "relevant", "small_talk", "hybrid"]:
        if data[cat]:
            pipeline.run_single_turn_test(cat, data[cat])

    pipeline.run_multi_turn_test(data["multi_turn"])

    # 5. Generate Report
    pipeline.calculate_metrics()