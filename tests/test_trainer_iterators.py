from training.trainer import _next_or_restart


def test_next_or_restart_cycles_exhausted_iterator() -> None:
    dataloader = ["first", "second"]
    iterator = iter(dataloader)

    first, iterator = _next_or_restart(dataloader, iterator)
    second, iterator = _next_or_restart(dataloader, iterator)
    restarted, iterator = _next_or_restart(dataloader, iterator)

    assert (first, second, restarted) == ("first", "second", "first")
