"""Small batching check for the T8 evaluator."""

from eval_task_policy import design_batches


def main():
    group = [(0, i) for i in range(8)]
    batches = design_batches(group, 4)
    assert len(batches) == 2
    assert all(len(batch) == 4 for batch in batches)
    assert [item for batch in batches for item in batch] == group


if __name__ == "__main__":
    main()
