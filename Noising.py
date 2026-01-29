from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Mathematics import DiscreteFunction

def saltAndPaperNoising (function: "DiscreteFunction", probability: float):
    import random

    for j in range(function.height):
        for i in range(function.width):
            if random.random() <= probability:
                function[i, j] = 255 * random.choice([0, 1])

    
def noising (function: "DiscreteFunction", minAdd: int, maxAdd: int):
    import random

    for j in range(function.height):
        for i in range(function.width):
            function[i, j] += random.randint(minAdd, maxAdd)

            if function[i, j] > 255:
                function[i, j] = 255

            if function[i, j] < 0:
                function[i, j] = 0