#include "draughts.h"

static int is_current_side_man(DraughtsPiece piece, DraughtsSide side)
{
    return (side == DRAUGHTS_BLACK && piece == DRAUGHTS_BLACK_MAN)
        || (side == DRAUGHTS_WHITE && piece == DRAUGHTS_WHITE_MAN);
}

static int append_quiet_move(DraughtsMove *moves, int max_moves, int *count, int from, int to)
{
    if (*count < max_moves) {
        moves[*count].path[0] = (unsigned char)from;
        moves[*count].path[1] = (unsigned char)to;
        moves[*count].path_len = 2;
        moves[*count].capture_count = 0;
    }

    ++(*count);
    return DRAUGHTS_OK;
}

static int try_add_man_quiet_move(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *count,
    int from,
    int row,
    int col
)
{
    int to;
    DraughtsPiece target;

    if (draughts_coord_to_square(row, col, &to) != DRAUGHTS_OK) {
        return DRAUGHTS_OK;
    }

    if (draughts_get_piece(state, to, &target) != DRAUGHTS_OK) {
        return DRAUGHTS_OK;
    }

    if (target == DRAUGHTS_EMPTY) {
        return append_quiet_move(moves, max_moves, count, from, to);
    }

    return DRAUGHTS_OK;
}

int draughts_generate_legal_moves(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *move_count
)
{
    DraughtsSide side;
    int count = 0;
    int square;

    if (state == NULL || move_count == NULL) {
        return DRAUGHTS_ERR_NULL;
    }
    if (max_moves < 0) {
        return DRAUGHTS_ERR_INVALID_ARGUMENT;
    }
    if (max_moves > 0 && moves == NULL) {
        return DRAUGHTS_ERR_NULL;
    }

    if (draughts_get_side_to_move(state, &side) != DRAUGHTS_OK) {
        return DRAUGHTS_ERR_INVALID_SIDE;
    }

    for (square = 1; square <= DRAUGHTS_SQUARE_COUNT; ++square) {
        DraughtsPiece piece;
        int row;
        int col;
        int forward_row;

        if (draughts_get_piece(state, square, &piece) != DRAUGHTS_OK) {
            continue;
        }
        if (!is_current_side_man(piece, side)) {
            continue;
        }
        if (draughts_square_to_coord(square, &row, &col) != DRAUGHTS_OK) {
            continue;
        }

        forward_row = row + (side == DRAUGHTS_BLACK ? 1 : -1);
        try_add_man_quiet_move(state, moves, max_moves, &count, square, forward_row, col - 1);
        try_add_man_quiet_move(state, moves, max_moves, &count, square, forward_row, col + 1);
    }

    *move_count = count;
    if (count > max_moves) {
        return DRAUGHTS_ERR_BUFFER_TOO_SMALL;
    }

    return DRAUGHTS_OK;
}
