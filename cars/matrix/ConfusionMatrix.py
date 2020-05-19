# Code provided by Allan Lochert
import numpy as np
from sty import fg
from typing import List


class ConfusionMatrix:

    def __init__(self, categories: List = []):
        self.categories = categories
        self.matrix = np.zeros((len(categories), len(categories)))

    def add(self, item):
        self.matrix[self.categories.index(item.category), self.categories.index(item.predicted_category)] += 1

    def add_pair(self, predicted_category, actual_category):
        self.matrix[self.categories.index(actual_category), self.categories.index(predicted_category)] += 1

    def __str__(self):
        result = f'          {fg.li_blue}'
        for i in range(0, len(self.categories)):
            result += f'{self.categories[i]:>8}'
        result += f'  accuracy{fg.rs}\n'
        all_total = 0
        for i in range(0, len(self.categories)):
            line = f'{fg.li_blue}{self.categories[i]:<10}{fg.rs}'
            total = 0
            for j in range(0, len(self.categories)):
                total += self.matrix[i, j]
                if i == j:
                    clr = fg(255)
                else:
                    clr = fg(248)
                line += f'{clr}{self.matrix[i, j]:8.0f}{fg.rs}'

            line += f'{fg.yellow}{self.matrix[i, i] / total * 100:9.1f}%{fg.rs}'
            result += f'{line}  \n'
            all_total += total

        total_correct = 0
        line = f'{fg.li_blue}Precision {fg.yellow}'
        for i in range(0, len(self.categories)):
            total = 0
            for j in range(0, len(self.categories)):
                total += self.matrix[j, i]
            total_correct += self.matrix[i, i]
            line += f'{self.matrix[i, i] / total * 100:7.1f}%'
        result += f'{line}'

        result += f'{fg(255)}{total_correct / all_total * 100:9.1f}%{fg.rs}'

        return result
