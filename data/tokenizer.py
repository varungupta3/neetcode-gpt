from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        chars = list(corpus)
        merges = []
        for _ in range(num_merges):
            # a. Count frequency of all adjacent token pairs
            freq = {}
            for pair in zip(chars[:-1], chars[1:]):
                freq[pair] = freq[pair] + 1 if pair in freq else 1
            # b. Find the most frequent pair (break ties lexicographically)
            max_count = max(freq.values())
            most_freq_pairs = [k for k in freq.keys() if freq[k]==max_count]
            most_freq_pair = sorted(most_freq_pairs)[0]
            print("most_freq_pair: ", most_freq_pair)
            # c. Merge all non-overlapping occurrences left to right
            new_chars = []
            i = 0
            while i < len(chars) - 1:
                print("while loop: ", i, len(chars))
                pair = (chars[i], chars[i+1])
                if pair == most_freq_pair:
                    new_chars.append(chars[i] + chars[i+1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            chars = new_chars
            # d. Record the merge as [token_a, token_b]
            merges.append([most_freq_pair[0], most_freq_pair[1]])
        return merges
