from itertools import combinations, product

# Given a list of player's cards and board cards, this function will generate all possible 3-card poker hands.
def generate_hand_combinations(my_cards: list[str], board_cards: list[str]) -> list[list[str]]:
    all_cards = my_cards + board_cards  # Combine player's cards and board cards
    all_combinations = list(combinations(all_cards, 3))  # Generate all 3-card combinations
    return all_combinations

# Define a function to categorize the hand combination
def categorize_hand_combination(combo: list[str]) -> str:
    # Sort the combo by rank
    sorted_combo = sorted(combo, key=lambda x: int(x[:-1]))
    ranks = [int(card[:-1]) for card in sorted_combo]
    suits = [card[-1] for card in sorted_combo]

    # Check for a straight flush
    if ranks == list(range(ranks[0], ranks[0] + 3)) and len(set(suits)) == 1:
        return 1
    
    # Check for trips
    elif ranks[0] == ranks[1] == ranks[2]:
        return 2
    
    # Check for a straight
    elif ranks == list(range(ranks[0], ranks[0] + 3)):
        return 3
    
    # Check for flush
    elif len(set(suits)) == 1:
        return 4
    
    # Check for a pair
    elif len(set(ranks)) == 2:
        return 5
    
    # Otherwise, it's a high card
    else:
        return 6

# Function to generate and categorize each combination
def generate_and_categorize_hands(my_cards: list[str], board_cards: list[str]) -> list[str]:
    hands = generate_hand_combinations(my_cards, board_cards)
    categorized_hands = [categorize_hand_combination(list(combo)) for combo in hands]
    return min(categorized_hands)

