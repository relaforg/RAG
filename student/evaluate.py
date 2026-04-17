import json
from student.models import StudentSearchResults, MinimalSource, MinimalSearchResults
from student.color import RESET, RED


class Evaluate:
    def overlap(self, src1: MinimalSource, src2: MinimalSource) -> float:
        overlap = max(
            min(src1.last_character_index, src2.last_character_index) -
            max(src1.first_character_index, src2.first_character_index),
            0)
        length = src2.last_character_index - src2.first_character_index
        return (overlap / length)

    def evaluate(self, student_answer_path: str, dataset_path: str) -> None:
        try:
            with open(student_answer_path, "r") as f:
                student_results = StudentSearchResults(**json.load(f))
        except (FileNotFoundError, PermissionError):
            print(
                f"{RED}{student_answer_path} does not exist or cannot be read{RESET}")
            return
        try:
            with open(dataset_path, "r") as f:
                raw_dataset = json.load(f)
        except (FileNotFoundError, PermissionError):
            print(f"{RED}{dataset_path} does not exist or cannot be read{RESET}")
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

        for i in [1, 3, 5, 10]:
            if (student_results.k < i):
                break
            self.compute_recall(
                student_results.search_results, ground_truth, i)

    def compute_recall(self, searches: list[MinimalSearchResults],
                       ground_truth: dict[str, list[MinimalSource]],
                       k: int) -> None:
        count = 0
        is_found = False
        for search in searches:
            for source in search.retrieved_sources[:k]:
                for truth in ground_truth.get(search.question_id, []):
                    if (self.overlap(source, truth)):
                        count += 1
                        is_found = True
                        break
                if (is_found):
                    is_found = False
                    break
        print(f"Recall@{k}: {count/len(searches)}")
