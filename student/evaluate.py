import json
from student.models import (StudentSearchResults,
                            MinimalSource,
                            MinimalSearchResults)
from student.color import RESET, RED


class Evaluate:
    """Evaluates retrieval quality using recall@k metrics."""

    def overlap(self, src1: MinimalSource, src2: MinimalSource) -> float:
        """Compute character overlap ratio between two sources.

        Args:
            src1: Retrieved source.
            src2: Ground truth source.

        Returns:
            Overlap ratio relative to src2 length, 0.0 if different files.
        """
        if (src1.file_path != src2.file_path):
            return (0)
        overlap = max(
            min(src1.last_character_index, src2.last_character_index) -
            max(src1.first_character_index, src2.first_character_index),
            0)
        length = src2.last_character_index - src2.first_character_index
        return (overlap / length)

    def evaluate(self, student_answer_path: str, dataset_path: str) -> None:
        """Load search results and ground truth, then print recall@k scores.

        Args:
            student_answer_path: Path to the student search results JSON.
            dataset_path: Path to the answered questions dataset JSON.
        """
        try:
            with open(student_answer_path, "r") as f:
                student_results = StudentSearchResults(**json.load(f))
        except (FileNotFoundError, PermissionError):
            print(f"{RED}{student_answer_path} does not "
                  f"exist or cannot be read{RESET}")
            return
        try:
            with open(dataset_path, "r") as f:
                raw_dataset = json.load(f)
        except (FileNotFoundError, PermissionError):
            print(f"{RED}{dataset_path} does not "
                  f"exist or cannot be read{RESET}")
            return

        ground_truth: dict[str, list[MinimalSource]] = {
            q["question_id"]: [
                MinimalSource(
                    file_path=s["file_path"],
                    text="",
                    first_character_index=s["first_character_index"],
                    last_character_index=s["last_character_index"]
                )
                for s in q["sources"]
            ]
            for q in raw_dataset["rag_questions"]
            if "sources" in q
        }

        for i in [1, 3, 5, 10, 15, 25]:
            if (student_results.k < i):
                break
            self.compute_recall(
                student_results.search_results, ground_truth, i)

    def compute_recall(self, searches: list[MinimalSearchResults],
                       ground_truth: dict[str, list[MinimalSource]],
                       k: int) -> None:
        """Compute and print recall@k over all questions.

        Args:
            searches: List of search results per question.
            ground_truth: Mapping from question_id to correct sources.
            k: Number of top retrieved sources to consider.
        """
        total_score = 0.0
        for search in searches:
            correct = ground_truth.get(search.question_id, [])
            if not correct:
                continue
            found = 0
            for truth in correct:
                for source in search.retrieved_sources[:k]:
                    if (self.overlap(source, truth) >= 0.05):
                        found += 1
                        break
            total_score += found / len(correct)
        print(f"Recall@{k}: {total_score / len(searches):.3f}")
