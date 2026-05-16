#include "draughts.h"

static int is_current_side_man(DraughtsPiece piece, DraughtsSide side)
{
    return (side == DRAUGHTS_BLACK && piece == DRAUGHTS_BLACK_MAN)
        || (side == DRAUGHTS_WHITE && piece == DRAUGHTS_WHITE_MAN);
}

static int is_opponent_piece(DraughtsPiece piece, DraughtsSide side)
{
    if (side == DRAUGHTS_BLACK) {
        return piece == DRAUGHTS_WHITE_MAN || piece == DRAUGHTS_WHITE_KING;
    }

    return piece == DRAUGHTS_BLACK_MAN || piece == DRAUGHTS_BLACK_KING;
}

static int is_captured_square(const DraughtsMove *move, int square)
{
    int i;

    for (i = 0; i < move->capture_count; ++i) {
        if (move->captures[i] == square) {
            return 1;
        }
    }

    return 0;
}

static DraughtsPiece get_virtual_piece(
    const DraughtsState *state,
    DraughtsPiece moving_piece,
    int origin,
    int current,
    int square
)
{
    DraughtsPiece piece;

    if (square == current) {
        return moving_piece;
    }
    if (square == origin) {
        return DRAUGHTS_EMPTY;
    }
    if (draughts_get_piece(state, square, &piece) != DRAUGHTS_OK) {
        return DRAUGHTS_EMPTY;
    }

    return piece;
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

static void append_best_capture_move(
    DraughtsMove *moves,
    int max_moves,
    int *count,
    int *best_capture_count,
    const DraughtsMove *candidate
)
{
    if (candidate->capture_count < *best_capture_count) {
        return;
    }

    if (candidate->capture_count > *best_capture_count) {
        *best_capture_count = candidate->capture_count;
        *count = 0;
    }

    if (*count < max_moves) {
        moves[*count] = *candidate;
    }
    ++(*count);
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

static void search_man_captures(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *count,
    int *best_capture_count,
    DraughtsSide side,
    DraughtsPiece moving_piece,
    int origin,
    int current,
    DraughtsMove *partial
)
{
    static const int row_delta[4] = { -1, -1, 1, 1 };
    static const int col_delta[4] = { -1, 1, -1, 1 };
    int current_row;
    int current_col;
    int found_next = 0;
    int direction;

    if (draughts_square_to_coord(current, &current_row, &current_col) != DRAUGHTS_OK) {
        return;
    }

    for (direction = 0; direction < 4; ++direction) {
        int middle_row = current_row + row_delta[direction];
        int middle_col = current_col + col_delta[direction];
        int landing_row = current_row + row_delta[direction] * 2;
        int landing_col = current_col + col_delta[direction] * 2;
        int middle_square;
        int landing_square;
        DraughtsPiece middle_piece;
        DraughtsPiece landing_piece;
        DraughtsMove next;

        if (draughts_coord_to_square(middle_row, middle_col, &middle_square) != DRAUGHTS_OK) {
            continue;
        }
        if (draughts_coord_to_square(landing_row, landing_col, &landing_square) != DRAUGHTS_OK) {
            continue;
        }
        if (is_captured_square(partial, middle_square)) {
            continue;
        }

        middle_piece = get_virtual_piece(state, moving_piece, origin, current, middle_square);
        landing_piece = get_virtual_piece(state, moving_piece, origin, current, landing_square);

        if (!is_opponent_piece(middle_piece, side) || landing_piece != DRAUGHTS_EMPTY) {
            continue;
        }
        if (partial->path_len >= DRAUGHTS_MAX_MOVE_PATH
            || partial->capture_count >= DRAUGHTS_MAX_MOVE_CAPTURES) {
            continue;
        }

        next = *partial;
        next.path[next.path_len] = (unsigned char)landing_square;
        ++next.path_len;
        next.captures[next.capture_count] = (unsigned char)middle_square;
        ++next.capture_count;

        found_next = 1;
        search_man_captures(
            state,
            moves,
            max_moves,
            count,
            best_capture_count,
            side,
            moving_piece,
            origin,
            landing_square,
            &next
        );
    }

    if (!found_next && partial->capture_count > 0) {
        append_best_capture_move(moves, max_moves, count, best_capture_count, partial);
    }
}

static void generate_man_capture_moves(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *count,
    int *best_capture_count,
    DraughtsSide side
)
{
    int square;

    for (square = 1; square <= DRAUGHTS_SQUARE_COUNT; ++square) {
        DraughtsPiece piece;
        DraughtsMove partial;

        if (draughts_get_piece(state, square, &piece) != DRAUGHTS_OK) {
            continue;
        }
        if (!is_current_side_man(piece, side)) {
            continue;
        }

        partial.path[0] = (unsigned char)square;
        partial.path_len = 1;
        partial.capture_count = 0;

        search_man_captures(
            state,
            moves,
            max_moves,
            count,
            best_capture_count,
            side,
            piece,
            square,
            square,
            &partial
        );
    }
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
    int best_capture_count = 0;
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

    generate_man_capture_moves(state, moves, max_moves, &count, &best_capture_count, side);
    if (count > 0) {
        *move_count = count;
        if (count > max_moves) {
            return DRAUGHTS_ERR_BUFFER_TOO_SMALL;
        }

        return DRAUGHTS_OK;
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
