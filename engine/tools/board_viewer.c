#include "draughts.h"

#include <stdio.h>
#include <string.h>

#define VIEWER_MAX_MOVES 256

typedef int (*ScenarioSetup)(DraughtsState *state);

typedef struct {
    const char *name;
    const char *description;
    ScenarioSetup setup;
} Scenario;

static int setup_initial(DraughtsState *state);
static int setup_man_capture(DraughtsState *state);
static int setup_man_multi(DraughtsState *state);
static int setup_king_quiet(DraughtsState *state);
static int setup_king_capture(DraughtsState *state);
static int setup_king_multi(DraughtsState *state);
static int setup_max_capture(DraughtsState *state);
static int setup_promotion(DraughtsState *state);
static int setup_terminal(DraughtsState *state);

static const Scenario scenarios[] = {
    { "initial", "standard initial position", setup_initial },
    { "man-capture", "mandatory man capture suppresses quiet moves", setup_man_capture },
    { "man-multi", "man multi-jump capture", setup_man_multi },
    { "king-quiet", "king long-range quiet movement", setup_king_quiet },
    { "king-capture", "king capture with multiple landing squares", setup_king_capture },
    { "king-multi", "king multi-jump capture", setup_king_multi },
    { "max-capture", "man and king captures share the maximum-capture rule", setup_max_capture },
    { "promotion", "black man can move to the back rank and promote when applied", setup_promotion },
    { "terminal", "current side has pieces but no legal move", setup_terminal }
};

static int scenario_count(void)
{
    return (int)(sizeof(scenarios) / sizeof(scenarios[0]));
}

static void print_usage(const char *program)
{
    printf("Usage:\n");
    printf("  %s --ids\n", program);
    printf("  %s --initial\n", program);
    printf("  %s --scenarios\n", program);
    printf("  %s --scenario NAME\n", program);
    printf("  %s --apply NAME\n", program);
    printf("  %s --play NAME\n", program);
    printf("  %s --script NAME N[,N...]\n", program);
    printf("  %s --all-scenarios\n", program);
}

static void print_id_board(void)
{
    int row;
    int col;
    int square;

    for (row = 0; row < DRAUGHTS_BOARD_ROWS; ++row) {
        for (col = 0; col < DRAUGHTS_BOARD_COLS; ++col) {
            if (draughts_coord_to_square(row, col, &square) == DRAUGHTS_OK) {
                printf("%3d", square);
            } else {
                printf("%3s", ".");
            }
        }
        printf("\n");
    }
}

static const char *side_name(DraughtsSide side)
{
    return side == DRAUGHTS_BLACK ? "black" : "white";
}

static const char *result_name(DraughtsGameResult result)
{
    switch (result) {
    case DRAUGHTS_GAME_ONGOING:
        return "ongoing";
    case DRAUGHTS_GAME_BLACK_WIN:
        return "black win";
    case DRAUGHTS_GAME_WHITE_WIN:
        return "white win";
    case DRAUGHTS_GAME_DRAW:
        return "draw";
    default:
        return "unknown";
    }
}

static void print_state_summary(const DraughtsState *state)
{
    DraughtsSide side;
    DraughtsGameResult result;

    if (draughts_get_side_to_move(state, &side) == DRAUGHTS_OK) {
        printf("side to move: %s\n", side_name(side));
    } else {
        printf("side to move: invalid\n");
    }

    if (draughts_get_game_result(state, &result) == DRAUGHTS_OK) {
        printf("game result: %s\n", result_name(result));
    } else {
        printf("game result: invalid\n");
    }
}

static void print_state_board(const DraughtsState *state)
{
    int row;
    int col;
    int square;
    DraughtsPiece piece;

    printf("    1   2   3   4   5   6   7   8   9  10\n");
    printf("  +---+---+---+---+---+---+---+---+---+---+\n");
    for (row = 0; row < DRAUGHTS_BOARD_ROWS; ++row) {
        printf("%2d|", row + 1);
        for (col = 0; col < DRAUGHTS_BOARD_COLS; ++col) {
            if (draughts_coord_to_square(row, col, &square) == DRAUGHTS_OK) {
                if (draughts_get_piece(state, square, &piece) == DRAUGHTS_OK) {
                    printf(" %c |", draughts_piece_to_char(piece));
                } else {
                    printf(" ? |");
                }
            } else {
                printf("   |");
            }
        }
        printf("\n");
        printf("  +---+---+---+---+---+---+---+---+---+---+\n");
    }

    printf("\n");
    printf("b = black man, B = black king\n");
    printf("w = white man, W = white king\n");
    printf(". = empty playable square, blank = non-playable square\n");
}

static void print_move(const DraughtsMove *move)
{
    int i;
    char sep = move->capture_count > 0 ? 'x' : '-';

    for (i = 0; i < move->path_len; ++i) {
        if (i > 0) {
            printf("%c", sep);
        }
        printf("%d", move->path[i]);
    }

    if (move->capture_count > 0) {
        printf(" captures");
        for (i = 0; i < move->capture_count; ++i) {
            printf("%s%d", i == 0 ? " " : ",", move->captures[i]);
        }
    }
}

static int print_legal_moves(const DraughtsState *state, DraughtsMove *moves, int *move_count)
{
    int result;
    int i;

    result = draughts_generate_legal_moves(state, moves, VIEWER_MAX_MOVES, move_count);
    if (result == DRAUGHTS_ERR_BUFFER_TOO_SMALL) {
        printf("legal moves: %d, viewer buffer too small\n", *move_count);
        return result;
    }
    if (result != DRAUGHTS_OK) {
        printf("legal moves: error %d\n", result);
        *move_count = 0;
        return result;
    }

    printf("legal moves: %d\n", *move_count);
    for (i = 0; i < *move_count; ++i) {
        printf("  %2d. ", i + 1);
        print_move(&moves[i]);
        printf("\n");
    }

    return DRAUGHTS_OK;
}

static int set_piece(DraughtsState *state, int square, DraughtsPiece piece)
{
    return draughts_set_piece(state, square, piece);
}

static int setup_initial(DraughtsState *state)
{
    return draughts_init(state) == DRAUGHTS_OK;
}

static int setup_man_capture(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 24, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_man_multi(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 21, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK
        && set_piece(state, 38, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_king_quiet(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK
        && set_piece(state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_king_capture(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK
        && set_piece(state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_king_multi(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK
        && set_piece(state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK
        && set_piece(state, 20, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_max_capture(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 22, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 28, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK
        && set_piece(state, 27, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK
        && set_piece(state, 37, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK
        && set_piece(state, 19, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_promotion(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 41, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static int setup_terminal(DraughtsState *state)
{
    if (draughts_clear(state) != DRAUGHTS_OK) {
        return 0;
    }
    return set_piece(state, 46, DRAUGHTS_BLACK_KING) == DRAUGHTS_OK
        && set_piece(state, 41, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 47, DRAUGHTS_BLACK_MAN) == DRAUGHTS_OK
        && set_piece(state, 1, DRAUGHTS_WHITE_MAN) == DRAUGHTS_OK;
}

static const Scenario *find_scenario(const char *name)
{
    int i;

    for (i = 0; i < scenario_count(); ++i) {
        if (strcmp(name, scenarios[i].name) == 0) {
            return &scenarios[i];
        }
    }

    return NULL;
}

static void print_scenarios(void)
{
    int i;

    for (i = 0; i < scenario_count(); ++i) {
        printf("%-14s %s\n", scenarios[i].name, scenarios[i].description);
    }
}

static int show_scenario(const Scenario *scenario, int apply_first_move)
{
    DraughtsState state;
    DraughtsMove moves[VIEWER_MAX_MOVES];
    int move_count = 0;
    int result;

    if (!scenario->setup(&state)) {
        printf("Failed to set up scenario: %s\n", scenario->name);
        return 1;
    }

    printf("== %s ==\n", scenario->name);
    printf("%s\n\n", scenario->description);
    print_state_summary(&state);
    print_state_board(&state);

    result = print_legal_moves(&state, moves, &move_count);
    if (result != DRAUGHTS_OK || !apply_first_move || move_count == 0) {
        return result == DRAUGHTS_OK || result == DRAUGHTS_ERR_BUFFER_TOO_SMALL ? 0 : 1;
    }

    printf("\napplying first move: ");
    print_move(&moves[0]);
    printf("\n\n");

    result = draughts_apply_move(&state, &moves[0]);
    if (result != DRAUGHTS_OK) {
        printf("apply failed: %d\n", result);
        return 1;
    }

    print_state_summary(&state);
    print_state_board(&state);
    print_legal_moves(&state, moves, &move_count);

    return 0;
}

static int show_named_scenario(const char *name, int apply_first_move)
{
    const Scenario *scenario = find_scenario(name);

    if (scenario == NULL) {
        printf("Unknown scenario: %s\n", name);
        print_scenarios();
        return 1;
    }

    return show_scenario(scenario, apply_first_move);
}

static int show_all_scenarios(void)
{
    int i;
    int failed = 0;

    for (i = 0; i < scenario_count(); ++i) {
        if (i > 0) {
            printf("\n");
        }
        if (show_scenario(&scenarios[i], 0) != 0) {
            failed = 1;
        }
    }

    return failed;
}

static int parse_positive_int(const char *text, int *value)
{
    int result = 0;
    int i;

    if (text == NULL || text[0] == '\0' || value == NULL) {
        return 0;
    }

    for (i = 0; text[i] != '\0'; ++i) {
        if (text[i] < '0' || text[i] > '9') {
            return 0;
        }
        result = result * 10 + (text[i] - '0');
    }

    if (result <= 0) {
        return 0;
    }

    *value = result;
    return 1;
}

static int show_turn(const DraughtsState *state, DraughtsMove *moves, int *move_count)
{
    int result;

    print_state_summary(state);
    print_state_board(state);
    result = print_legal_moves(state, moves, move_count);
    if (result != DRAUGHTS_OK) {
        return result == DRAUGHTS_ERR_BUFFER_TOO_SMALL ? 0 : 1;
    }

    return 0;
}

static int apply_indexed_move(DraughtsState *state, const DraughtsMove *moves, int move_count, int move_number)
{
    int result;

    if (move_number < 1 || move_number > move_count) {
        printf("invalid move number: %d\n", move_number);
        return 1;
    }

    printf("applying move %d: ", move_number);
    print_move(&moves[move_number - 1]);
    printf("\n\n");

    result = draughts_apply_move(state, &moves[move_number - 1]);
    if (result != DRAUGHTS_OK) {
        printf("apply failed: %d\n", result);
        return 1;
    }

    return 0;
}

static int play_scenario(const Scenario *scenario)
{
    DraughtsState state;
    DraughtsMove moves[VIEWER_MAX_MOVES];
    int move_count = 0;
    char line[64];

    if (!scenario->setup(&state)) {
        printf("Failed to set up scenario: %s\n", scenario->name);
        return 1;
    }

    printf("== interactive: %s ==\n", scenario->name);
    printf("%s\n", scenario->description);
    printf("commands: move number, r = redraw, q = quit\n\n");

    while (1) {
        int move_number;

        if (show_turn(&state, moves, &move_count) != 0) {
            return 1;
        }
        if (move_count == 0) {
            printf("no legal moves; game over\n");
            return 0;
        }

        printf("\nmove> ");
        if (fgets(line, sizeof(line), stdin) == NULL) {
            printf("\nend of input\n");
            return 0;
        }
        line[strcspn(line, "\r\n")] = '\0';

        if (strcmp(line, "q") == 0 || strcmp(line, "quit") == 0) {
            printf("quit\n");
            return 0;
        }
        if (strcmp(line, "r") == 0 || strcmp(line, "redraw") == 0 || line[0] == '\0') {
            printf("\n");
            continue;
        }
        if (!parse_positive_int(line, &move_number)) {
            printf("unknown command: %s\n\n", line);
            continue;
        }
        if (apply_indexed_move(&state, moves, move_count, move_number) != 0) {
            printf("\n");
            continue;
        }
    }
}

static int play_named_scenario(const char *name)
{
    const Scenario *scenario = find_scenario(name);

    if (scenario == NULL) {
        printf("Unknown scenario: %s\n", name);
        print_scenarios();
        return 1;
    }

    return play_scenario(scenario);
}

static int script_scenario(const Scenario *scenario, const char *sequence)
{
    DraughtsState state;
    DraughtsMove moves[VIEWER_MAX_MOVES];
    int move_count = 0;
    const char *cursor = sequence;
    int step = 1;

    if (!scenario->setup(&state)) {
        printf("Failed to set up scenario: %s\n", scenario->name);
        return 1;
    }

    printf("== script: %s ==\n", scenario->name);
    printf("%s\n\n", scenario->description);

    while (cursor != NULL && cursor[0] != '\0') {
        char token[32];
        int token_len = 0;
        int move_number;
        int result;

        while (cursor[token_len] != '\0' && cursor[token_len] != ',') {
            if (token_len + 1 >= (int)sizeof(token)) {
                printf("script token too long at step %d\n", step);
                return 1;
            }
            token[token_len] = cursor[token_len];
            ++token_len;
        }
        token[token_len] = '\0';

        result = draughts_generate_legal_moves(&state, moves, VIEWER_MAX_MOVES, &move_count);
        if (result != DRAUGHTS_OK) {
            printf("legal move generation failed at step %d: %d\n", step, result);
            return 1;
        }
        if (!parse_positive_int(token, &move_number)) {
            printf("invalid script token at step %d: %s\n", step, token);
            return 1;
        }

        printf("step %d, legal moves: %d\n", step, move_count);
        if (apply_indexed_move(&state, moves, move_count, move_number) != 0) {
            return 1;
        }

        cursor += token_len;
        if (cursor[0] == ',') {
            ++cursor;
        }
        ++step;
    }

    print_state_summary(&state);
    print_state_board(&state);
    print_legal_moves(&state, moves, &move_count);

    return 0;
}

static int script_named_scenario(const char *name, const char *sequence)
{
    const Scenario *scenario = find_scenario(name);

    if (scenario == NULL) {
        printf("Unknown scenario: %s\n", name);
        print_scenarios();
        return 1;
    }

    return script_scenario(scenario, sequence);
}

int main(int argc, char **argv)
{
    DraughtsState state;

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--ids") == 0 && argc == 2) {
        print_id_board();
        return 0;
    }

    if (strcmp(argv[1], "--initial") == 0 && argc == 2) {
        if (draughts_init(&state) != DRAUGHTS_OK) {
            printf("Failed to initialize board\n");
            return 1;
        }
        print_state_summary(&state);
        print_state_board(&state);
        return 0;
    }

    if (strcmp(argv[1], "--scenarios") == 0 && argc == 2) {
        print_scenarios();
        return 0;
    }

    if (strcmp(argv[1], "--scenario") == 0 && argc == 3) {
        return show_named_scenario(argv[2], 0);
    }

    if (strcmp(argv[1], "--apply") == 0 && argc == 3) {
        return show_named_scenario(argv[2], 1);
    }

    if (strcmp(argv[1], "--play") == 0 && argc == 3) {
        return play_named_scenario(argv[2]);
    }

    if (strcmp(argv[1], "--script") == 0 && argc == 4) {
        return script_named_scenario(argv[2], argv[3]);
    }

    if (strcmp(argv[1], "--all-scenarios") == 0 && argc == 2) {
        return show_all_scenarios();
    }

    print_usage(argv[0]);
    return 1;
}
