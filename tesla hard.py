'''
Tesla Autopilot Round-1 (2024/09/17)
Please write a function that solves the game of SET!

SET! is a card game where the objective is to find all the "sets" among a set of n cards.

Each card has 4 attributes:

1. Shape  --  S
2. Color  --  C
3. Number --  N
4. Fill   --  F

Each attribute has only 3 possible values (0, 1, 2). For example, color could be one of Red, Green, or Blue.

Note: Each card is unique.

A "set" is defined as 3 cards where each ATTRIBUTE is either the same or different.

Ex. "A valid set"
    S  C  N  F
C1  0, 1, 2, 0
C2  0, 0, 0, 1
C3  0, 2, 1, 2

Ex. "An invalid set"  
    S  C  N  F
C1  0, 1, 2, 0
C2  0, 0, 0, 1
C3  0, 2, 1, 1

- If c1 and c2 have the same value, the third card must also have that value.
- If c1 and c2 have different values, the third card must have the remaining value.
'''

def find_sets(cards): 
    cards = list(tuple(card) for card in cards)
    print(cards)
    card_set = set(cards)
    print(card_set)
    res = set() 

    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            c1, c2 = cards[i], cards[j]
            c3 = third_card(c1, c2)

            if c3 in card_set:
                valid_set = tuple(sorted([c1, c2, c3])) 
                res.add(valid_set)
    return res
  
  
def third_card(c1, c2): 
    c3 = []

    for i in range(4):
        if c1[i] == c2[i]:
            c3.append(c1[i])
        else:
            c3.append(3 - c1[i] - c2[i])

    return tuple(c3)


# Test
cards = [
    [0, 1, 2, 0],  # Card 1      this 
    [0, 0, 0, 1],  # Card 2      this
    [0, 2, 1, 1],  # Card 3
    [2, 2, 2, 2],  # Card 4
    [1, 1, 1, 1],  # Card 5
    [0, 2, 1, 2],  # Card 6      this
]

sets = find_sets(cards)
print(type(sets))
for s in sets:
    print(s)
    print(type(s))


'''
list 不能放入set，tuple可以

list 不能hash，放入set里前要转为tuple

cards = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# 你可以将列表转换为元组再放入集合
card_set = set(tuple(card) for card in cards)

还有一个不同就是tuple不可变，sort出来都是list，除此意外string也不可变


[[list(value) for value in x] for x in res]才能return出来的都是是双[[]]
'''