from src.players.cfr import FullCFRPlayer


class BestPlayer1(FullCFRPlayer):
    """Submission wrapper for the strongest overall agent."""

    def __init__(self, player_idx):
        super().__init__(
            player_idx=player_idx,
            search_time_limit=0.82,
            max_samples=6,
            base_iterations=8,
            endgame_iterations=28,
            max_profile_work=50000,
            min_iterations=4,
            max_cfr_hand_size=6,
            n_cards=104,
            board_size_x=5,
            rank_weight=0.35,
            dump_weight=0.15,
        )
