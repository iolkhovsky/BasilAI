class RandomAccessQueue:
    def __init__(self, max_size=None):
        self._items = []
        self._max_size = max_size

    def push(self, item):
        if self.max_size is not None and (len(self) + 1) > self.max_size:
            self.pop()
        self._items.append(item)

    def pop(self):
        return self._items.pop(0)

    def __getitem__(self, index):
        return self.items[len(self) - 1 - index]

    def __len__(self):
        return len(self.items)
