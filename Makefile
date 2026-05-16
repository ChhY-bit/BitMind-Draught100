CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -Werror -Iengine/include

BUILD_DIR := build
ENGINE_OBJS := $(BUILD_DIR)/board.o
MOVE_OBJS := $(BUILD_DIR)/move.o
ALL_ENGINE_OBJS := $(ENGINE_OBJS) $(MOVE_OBJS)
TEST_BOARD := $(BUILD_DIR)/test_board
TEST_MOVES := $(BUILD_DIR)/test_moves
BOARD_VIEWER := $(BUILD_DIR)/board_viewer

.PHONY: all test tools clean

all: test tools

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/board.o: engine/src/board.c engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c engine/src/board.c -o $@

$(BUILD_DIR)/move.o: engine/src/move.c engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c engine/src/move.c -o $@

$(TEST_BOARD): engine/tests/test_board.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tests/test_board.c $(ALL_ENGINE_OBJS) -o $@

$(TEST_MOVES): engine/tests/test_moves.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tests/test_moves.c $(ALL_ENGINE_OBJS) -o $@

$(BOARD_VIEWER): engine/tools/board_viewer.c $(ALL_ENGINE_OBJS) engine/include/draughts.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) engine/tools/board_viewer.c $(ALL_ENGINE_OBJS) -o $@

test: $(TEST_BOARD) $(TEST_MOVES)
	$(TEST_BOARD)
	$(TEST_MOVES)

tools: $(BOARD_VIEWER)

clean:
	rm -rf $(BUILD_DIR)
