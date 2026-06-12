#ifndef DRAUGHTS_H
#define DRAUGHTS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DRAUGHTS_BOARD_ROWS 10
#define DRAUGHTS_BOARD_COLS 10
#define DRAUGHTS_SQUARE_COUNT 50
#define DRAUGHTS_MAX_MOVE_PATH 32
#define DRAUGHTS_MAX_MOVE_CAPTURES 20

static inline int is_valid_square_id(int square_id)
{
    return square_id >= 1 && square_id <= DRAUGHTS_SQUARE_COUNT;
}

typedef enum {
    DRAUGHTS_OK = 0,
    DRAUGHTS_ERR_NULL = -1,
    DRAUGHTS_ERR_INVALID_SQUARE = -2,
    DRAUGHTS_ERR_INVALID_COORD = -3,
    DRAUGHTS_ERR_INVALID_PIECE = -4,
    DRAUGHTS_ERR_INVALID_SIDE = -5,
    DRAUGHTS_ERR_BUFFER_TOO_SMALL = -6,
    DRAUGHTS_ERR_INVALID_ARGUMENT = -7,
    DRAUGHTS_ERR_INVALID_MOVE = -8
} DraughtsResult;

typedef enum {
    DRAUGHTS_EMPTY = 0,
    DRAUGHTS_BLACK_MAN = 1,
    DRAUGHTS_BLACK_KING = 2,
    DRAUGHTS_WHITE_MAN = 3,
    DRAUGHTS_WHITE_KING = 4
} DraughtsPiece;

typedef enum {
    DRAUGHTS_BLACK = 0,
    DRAUGHTS_WHITE = 1
} DraughtsSide;

typedef enum {
    DRAUGHTS_GAME_ONGOING = 0,
    DRAUGHTS_GAME_BLACK_WIN = 1,
    DRAUGHTS_GAME_WHITE_WIN = 2,
    DRAUGHTS_GAME_DRAW = 3
} DraughtsGameResult;

typedef struct {
    unsigned char squares[DRAUGHTS_SQUARE_COUNT];
    unsigned char side_to_move;
} DraughtsState;

typedef struct {
    unsigned char path[DRAUGHTS_MAX_MOVE_PATH];
    unsigned char path_len;
    unsigned char captures[DRAUGHTS_MAX_MOVE_CAPTURES];
    unsigned char capture_count;
} DraughtsMove;

/*
 * Input: row and col are zero-based 10x10 board coordinates.
 * Output: returns 1 when the coordinate is a playable dark square, otherwise 0.
 */
int draughts_is_playable_coord(int row, int col);

/*
 * Input: square_id is the public one-based playable square id, from 1 to 50.
 * Output: writes zero-based row and col when requested.
 */
int draughts_square_to_coord(int square_id, int *row, int *col);

/*
 * Input: row and col are zero-based 10x10 board coordinates.
 * Output: writes the public one-based square id for playable coordinates.
 */
int draughts_coord_to_square(int row, int col, int *square_id);

/*
 * Input: state points to writable storage.
 * Output: initializes the standard starting position with black to move.
 */
int draughts_init(DraughtsState *state);

/*
 * Input: state points to writable storage.
 * Output: clears all playable squares and sets black to move.
 */
int draughts_clear(DraughtsState *state);

/*
 * Input: dst and src point to valid state storage.
 * Output: copies the complete state from src to dst.
 */
int draughts_copy(DraughtsState *dst, const DraughtsState *src);

/*
 * Input: state is a valid board state, square_id is 1..50.
 * Output: writes the piece currently on that square.
 */
int draughts_get_piece(const DraughtsState *state, int square_id, DraughtsPiece *piece);

/*
 * Input: state is a valid board state, square_id is 1..50, piece is a DraughtsPiece.
 * Output: stores the piece on that square.
 */
int draughts_set_piece(DraughtsState *state, int square_id, DraughtsPiece piece);

/*
 * Input: state is a valid board state.
 * Output: returns the side to move.
 */
int draughts_get_side_to_move(const DraughtsState *state, DraughtsSide *side);

/*
 * Input: state is a valid board state and side is a DraughtsSide.
 * Output: updates the side to move.
 */
int draughts_set_side_to_move(DraughtsState *state, DraughtsSide side);

/*
 * Input: piece is a DraughtsPiece value.
 * Output: returns a compact printable character for text board views.
 */
char draughts_piece_to_char(DraughtsPiece piece);

/*
 * Input: state is a valid board state, moves points to caller storage with max_moves entries.
 * Output: generates current-side legal moves into moves and writes the total move count.
 */
int draughts_generate_legal_moves(
    const DraughtsState *state,
    DraughtsMove *moves,
    int max_moves,
    int *move_count
);

/*
 * Input: state is a valid board state and move describes a candidate move.
 * Output: writes 1 when move exactly matches one current legal move, otherwise 0.
 */
int draughts_is_legal_move(
    const DraughtsState *state,
    const DraughtsMove *move,
    int *is_legal
);

/*
 * Input: state is a valid board state and move is a current legal move.
 * Output: applies the move, removes captured pieces, promotes men on final back rank, and switches side.
 */
int draughts_apply_move(DraughtsState *state, const DraughtsMove *move);

/*
 * Input: state is a valid board state.
 * Output: writes the current game result based on material and current-side legal moves.
 */
int draughts_get_game_result(const DraughtsState *state, DraughtsGameResult *result);

#ifdef __cplusplus
}
#endif

#endif
