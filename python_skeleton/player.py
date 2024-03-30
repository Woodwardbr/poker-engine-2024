"""
Simple example pokerbot, written in Python.
"""

import random
import torch
import numpy as np
from python_skeleton.gae.policies.MLP_policy import MLPPolicyAC
from python_skeleton.gae.infrastructure.pytorch_util import build_mlp
import pickle
import itertools
from typing import Optional
from python_skeleton.skeleton.actions import Action, CallAction, CheckAction, FoldAction, RaiseAction
from python_skeleton.skeleton.states import GameState, TerminalState, RoundState
from python_skeleton.skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from python_skeleton.skeleton.bot import Bot
from python_skeleton.skeleton.runner import parse_args, run_bot
from python_skeleton.skeleton.evaluate import evaluate


def card_to_int(card: str):
    rank, suit = card[0], card[1]
    suit = {"s": 0, "h": 1, "d": 2}[suit]
    return (suit * 10 + int(rank))




class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self) -> None:
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        self.policy = MLPPolicyAC(
            ac_dim=2,
            ob_dim=16,
            n_layers=2,
            size=64,
            learning_rate=5e-3,
        )

        self.policy.action_mlp.load_state_dict(torch.load('action_model.pth', map_location=torch.device('cpu')))
        self.policy.bet_mlp.load_state_dict(torch.load('bet_model.pth', map_location=torch.device('cpu')))

        self.log = []
        #self.pre_computed_probs = pickle.load(open("python_skeleton/skeleton/pre_computed_probs.pkl", "rb"))
        pass

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int) -> None:
        """
        Called when a new round starts. Called NUM_ROUNDS times.
        
        Args:
            game_state (GameState): The state of the game.
            round_state (RoundState): The state of the round.
            active (int): Your player's index.

        Returns:
            None
        """
        #my_bankroll = game_state.bankroll # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        #game_clock = game_state.game_clock # the total number of seconds your bot has left to play this game
        #round_num = game_state.round_num # the round number from 1 to NUM_ROUNDS
        #my_cards = round_state.hands[0] # your cards
        #big_blind = bool(active) # True if you are the big blind
        self.log = []
        self.log.append("================================")
        self.log.append("new round")
        pass

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int, is_match_over: bool) -> Optional[str]:
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Args:
            game_state (GameState): The state of the game.
            terminal_state (TerminalState): The state of the round when it ended.
            active (int): Your player's index.

        Returns:
            Your logs.
        """
        #my_delta = terminal_state.deltas[active] # your bankroll change from this round
        #previous_state = terminal_state.previous_state # RoundState before payoffs
        #street = previous_state.street # 0, 3, 4, or 5 representing when this round ended
        #my_cards = previous_state.hands[0] # your cards
        #opp_cards = previous_state.hands[1] # opponent's cards or [] if not revealed
        self.log.append("game over")
        self.log.append("================================\n")

        return self.log


    def dict_obs_to_np_obs(self, dict_obs):
        obs_arr = []
        legal_act_arr = [0,0,0,0]
        if isinstance(dict_obs["legal_actions"], np.ndarray):
            obs_arr.extend(dict_obs["legal_actions"])
        else:
            obs_arr.extend(np.array([int(action in dict_obs["legal_actions"]) for action in [FoldAction, CallAction, CheckAction, RaiseAction]]).astype(np.int8))
        obs_arr.append(dict_obs['street'])
        for i in range(len(dict_obs["my_cards"])):
            if isinstance(dict_obs["my_cards"][i], str):
                obs_arr.append(card_to_int(dict_obs["my_cards"][i]))
            else:
                obs_arr.append(dict_obs["my_cards"][i])
        
        for i in range(len(dict_obs["board_cards"])):
            if isinstance(dict_obs["board_cards"][i], str):
                obs_arr.append(card_to_int(dict_obs["board_cards"][i]))
            else:
                obs_arr.append(dict_obs["board_cards"][i])

        for i in range(2-len(dict_obs["board_cards"])):
            obs_arr.append(0)
        if isinstance(dict_obs["my_pip"], np.ndarray):
            obs_arr.append(dict_obs["my_pip"].squeeze())
            obs_arr.append(dict_obs["opp_pip"].squeeze())
            obs_arr.append(dict_obs["my_stack"].squeeze())
            obs_arr.append(dict_obs["opp_stack"].squeeze())
            obs_arr.append(dict_obs["my_bankroll"].squeeze())
            obs_arr.append(dict_obs["min_raise"].squeeze())
            obs_arr.append(dict_obs["max_raise"].squeeze())
        else:
            obs_arr.append(dict_obs["my_pip"])
            obs_arr.append(dict_obs["opp_pip"])
            obs_arr.append(dict_obs["my_stack"])
            obs_arr.append(dict_obs["opp_stack"])
            obs_arr.append(dict_obs["my_bankroll"])
            obs_arr.append(dict_obs["min_raise"])
            obs_arr.append(dict_obs["max_raise"])
        return np.array(obs_arr)
    
    def get_action_pair(self, observation: dict):
        obs_arr = self.dict_obs_to_np_obs(observation)
        return self.policy.get_action(obs_arr).astype(int)


    def get_action(self, observation: dict) -> Action:
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Args:
            observation (dict): The observation of the current state.
            {
                "legal_actions": List of the Actions that are legal to take.
                "street": 0, 1, or 2 representing pre-flop, flop, or river respectively
                "my_cards": List[str] of your cards, e.g. ["1s", "2h"]
                "board_cards": List[str] of the cards on the board
                "my_pip": int, the number of chips you have contributed to the pot this round of betting
                "opp_pip": int, the number of chips your opponent has contributed to the pot this round of betting
                "my_stack": int, the number of chips you have remaining
                "opp_stack": int, the number of chips your opponent has remaining
                "my_bankroll": int, the number of chips you have won or lost from the beginning of the game to the start of this round
                "min_raise": int, the smallest number of chips for a legal bet/raise
                "max_raise": int, the largest number of chips for a legal bet/raise
            }

        Returns:
            Action: The action you want to take.
        """
        my_contribution = STARTING_STACK - observation["my_stack"] # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - observation["opp_stack"] # the number of chips your opponent has contributed to the pot
        continue_cost = observation["opp_pip"] - observation["my_pip"] # the number of chips needed to stay in the pot

        self.log.append("My cards: " + str(observation["my_cards"]))
        self.log.append("Board cards: " + str(observation["board_cards"]))
        self.log.append("My stack: " + str(observation["my_stack"]))
        self.log.append("My contribution: " + str(my_contribution))
        self.log.append("My bankroll: " + str(observation["my_bankroll"]))
        obs_arr = self.dict_obs_to_np_obs(observation)

        policy_action = self.policy.get_action(obs_arr).astype(int)
        self.log.append("Obs Arr:" + str(obs_arr))
        self.log.append("Action pair" + str(policy_action))

        match policy_action[0]:
            case 0:
                return FoldAction()
            case 1:
                return CallAction()
            case 2:
                return CheckAction()
            case 3:
                return RaiseAction(policy_action[1])

        return FoldAction()

if __name__ == '__main__':
    run_bot(Player(), parse_args())