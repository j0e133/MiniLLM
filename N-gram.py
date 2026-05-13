
from unidecode import unidecode

import math, os, random, statistics

from typing import Optional, Self, Iterable
from json import loads, dumps



def clean_file(filename: str) -> None:
    with open(filename, 'r', encoding="utf8") as f:
        data = unidecode(f.read()).replace('_', '')
    
    with open(filename, 'w') as f:
        f.write(data)


def softmax(values: Iterable[float], temperature: float) -> list[float]:
    inv_temp = 1 / temperature

    exps = [math.exp(val * inv_temp) for val in values]

    inv_total = 1 / sum(exp for exp in exps)

    return [exp * inv_total for exp in exps]



class NgramModel:
    def __init__(self, n: int, sep: Optional[str] = None):
        self.n = n - 1
        self.sep = sep

        self.next_counts: dict[str, dict[str, int]] = {}

    @property
    def num_tokens(self) -> int:
        return sum(sum(count for count in counts.values()) for counts in self.next_counts.values())

    @property
    def num_grams(self) -> int:
        return len(self.next_counts)

    @property
    def quartiles(self) -> list[float]:
        lengths = [len(counts) for counts in self.next_counts.values()]

        return [float(min(lengths)), *statistics.quantiles(lengths), float(max(lengths))]

    def split(self, string: str) -> list[str]:
        match self.sep:
            case None:
                return list(string)

            case '':
                return string.split()

            case sep:
                return list(filter(lambda s: len(s) != 0, string.split(sep)))

    def join(self, grams: list[str]) -> str:
        match self.sep:
            case None:
                return ''.join(grams)

            case '':
                return ' '.join(grams)

            case sep:
                return sep.join(grams)

    def add_from_dir(self, dir: str) -> None:
        for file in os.listdir(dir):
            path = f"{dir}/{file}"

            if os.path.isfile(path):
                clean_file(path)
                self.add_from_file(path)

    def add_from_file(self, filename: str) -> None:
        with open(filename, encoding="utf8") as f:
            grams = self.split(f.read().lower())

        a = grams[:self.n]
        b = grams[self.n]
        i = self.n + 1

        while i < len(grams):
            self.add_next(self.join(a), b)

            a = a[1:] + [b]
            b = grams[i]
            i += 1

    def add_next(self, a: str, b: str) -> None:
        if a not in self.next_counts:
            self.next_counts[a] = {}

        first = self.next_counts[a]

        if b not in first:
            first[b] = 1
        else:
            first[b] += 1

    def calculate_probs(self, gram: str) -> dict[str, float]:
        freqs = self.next_counts[gram]
        inv_total = 1 / sum(count for count in freqs.values())

        probs = {gram: count * inv_total for gram, count in zip(freqs.keys(), freqs.values())}

        return probs

    def generate(self, text: str, num_grams: int, temperature: float = 1.0) -> str:
        grams = self.split(text.lower())
        next_probs = {}

        for _ in range(num_grams):
            gram = self.join(grams[-self.n:])

            if gram not in next_probs:
                next_probs[gram] = self.calculate_probs(gram)

            next_grams = list(next_probs[gram].keys())
            probs = softmax(next_probs[gram].values(), temperature)

            next_gram = random.choices(next_grams, probs)[0]
            grams.append(next_gram)

        return self.join(grams)

    @classmethod
    def load(cls, save_name: str) -> Self:
        with open(f"saves/{save_name}.model", 'r') as f:
            data = loads(f.read())

            n = data["n"]
            sep = data["sep"]
            next_counts = data["next_counts"]

            model = cls(n, sep)
            model.next_counts = next_counts

            return model

    def save(self, save_name: str):
        with open(f"saves/{save_name}.model", "w") as f:
            f.write(dumps(
                {
                    "n": self.n,
                    "sep": self.sep,
                    "next_counts": self.next_counts
                }
            ))


class NgramWordModel(NgramModel):
    def __init__(self, n: int):
        super().__init__(n, ' ')


class BigramModel(NgramModel):
    def __init__(self, sep: Optional[str] = None):
        super().__init__(2, sep)


class BigramWordModel(BigramModel):
    def __init__(self):
        super().__init__(' ')



if __name__ == "__main__":
    model = NgramWordModel(4)
    model.add_from_dir("books")

    print(f"Stats for model:")
    print(f"Training \"tokens\": {model.num_tokens:,}")
    print(f"Number of n-grams: {model.num_grams:,}")
    print(f"Quartiles for continuations: {model.quartiles}")
    print()

    print(model.generate("A rose by any other name", 1000, 0.5))

