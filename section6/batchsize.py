def get_iteration_count(train_data_len: int, batch_size: int) -> int:
    if train_data_len % batch_size == 0:
        return train_data_len // batch_size
    else:
        return train_data_len // batch_size + 1

def solution():
    train_data_len, batch_size = map(int, input().split())
    n_iterations = get_iteration_count(train_data_len, batch_size)
    print(n_iterations)

solution()