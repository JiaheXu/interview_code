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
    cards = tuple(tuple(card) for card in cards)
    print(cards)
    card_set = set(cards)
    print(cards)
    res = set()

    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            third = third_card(cards[i], cards[j])
            if third in card_set:
                res.add(tuple(sorted((cards[i], cards[j], third))))
    return res



def third_card(c1, c2):
    thild = [0] * 4
    for i in range(4):
        if c1[i] == c2[i]:
            thild[i] = c1[i]
        else:
            thild[i] = 3 - c1[i] - c2[i]
    return tuple(thild)


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
for s in sets:
    print(s)